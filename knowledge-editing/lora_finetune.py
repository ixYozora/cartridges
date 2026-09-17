"""
LoRA Fine-tuning Script for Knowledge Editing
Fine-tunes Qwen3-4b on self-study generated dataset using LoRA.
Uses torch.compile and bfloat16 for efficiency.

Usage:
    python lora_finetune.py \
        --parquet-path /path/to/dataset.parquet \
        --output-dir ./lora_output \
        --batch-size 4 \
        --num-epochs 3 \
        --learning-rate 2e-4

Example:
    python lora_finetune.py \
        --parquet-path /home/imasoudian/cartridges/outputs/2026-01-13-01-18-12-synthesize/WholeAKEW-0/artifact/dataset.parquet \
        --output-dir ./lora_qwen3_4b \
        --batch-size 4 \
        --gradient-accumulation-steps 8 \
        --num-epochs 3 \
        --learning-rate 2e-4 \
        --save-steps 500
"""

import os
import json
import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import argparse
import wandb

from cartridges.structs import Conversation, read_conversations

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Set CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_conversations_from_parquet(parquet_path: str) -> List[Conversation]:
    """Load conversations from parquet file."""
    return read_conversations(parquet_path)


def build_training_example(conversation: Conversation, tokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    """Tokenize one conversation with assistant-only loss masking.

    The full conversation is rendered with the chat template, then the loss is
    masked (label = -100) on every token that is not part of the final assistant
    turn. The prompt rendered with add_generation_prompt=True is a strict prefix
    of the full text, so its token length marks where supervision begins.
    Padding is added later by the collator and is also masked, so no gradient
    signal comes from prompt or padding tokens.
    """
    messages = [msg.to_message_dict() for msg in conversation.messages]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_length]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = list(input_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def prepare_dataset(parquet_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Prepare dataset from parquet file for training (assistant-only labels)."""
    print(f"Loading conversations from {parquet_path}...")
    conversations = load_conversations_from_parquet(parquet_path)
    print(f"Loaded {len(conversations)} conversations")

    examples = []
    skipped = 0
    for conv in conversations:
        example = build_training_example(conv, tokenizer, max_length)
        # Drop convos with no supervised tokens (e.g. prompt already >= max_length).
        if all(label == -100 for label in example["labels"]):
            skipped += 1
            continue
        examples.append(example)

    print(f"Prepared {len(examples)} training examples ({skipped} skipped: no assistant tokens)")
    return Dataset.from_list(examples)


def create_lora_model(model, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Create LoRA model from base model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class KESets:
    """Forget and retain sequences for the 3-term objective, tokenized once.

    forget  each edit's own question (and its paraphrases) completed with the OLD answer --
            the statement the model must stop making. The 2026-08 attempt looked for the old
            answer inside the Self-Study targets instead and found almost nothing (the
            fidelity filter had already removed those rows), so its forget term never fired.
    retain  CounterFact neighbourhood prompts completed with THEIR correct answer, which is
            the same old value carried by a DIFFERENT subject. That is exactly the pair plain
            finetuning confuses (locality drops ~10 points, 2026-09-17), so the KL term
            anchors it against the un-finetuned model.

    Sequences are padded to one fixed width so the extra forward passes do not make
    torch.compile re-trace every step.
    """

    def __init__(self, path, tokenizer, max_edits=0, retain_per_edit=2, width=48):
        entries = json.loads(Path(path).read_text())
        if max_edits:
            entries = entries[:max_edits]
        self.width = width
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.forget, self.retain = [], []
        for e in entries:
            rw = e["requested_rewrite"]
            subject = rw["subject"]
            old = " " + rw["target_true"]["str"]
            prompts = [rw["prompt"].format(subject)]
            prompts += [p.replace("{}", subject) if "{}" in p else p
                        for p in e.get("paraphrase_prompts", [])]
            for p in prompts:
                seq = self._tokenize(tokenizer, p, old)
                if seq:
                    self.forget.append(seq)
            for p in e.get("neighborhood_prompts", [])[:retain_per_edit]:
                seq = self._tokenize(tokenizer, p, old)
                if seq:
                    self.retain.append(seq)
        print(f"KE sets: {len(self.forget)} forget sequences, {len(self.retain)} retain "
              f"sequences from {len(entries)} edits")

    def _tokenize(self, tokenizer, prompt, answer):
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        ids = tokenizer(prompt + answer, add_special_tokens=True)["input_ids"]
        if len(ids) <= len(prompt_ids) or len(ids) > self.width:
            return None
        return ids, len(prompt_ids)

    def batch(self, which, size, rng, device):
        """-> input_ids, attention_mask, target_mask (positions predicting the answer)."""
        pool = self.forget if which == "forget" else self.retain
        picks = [pool[rng.randrange(len(pool))] for _ in range(min(size, len(pool)))]
        ids = torch.full((len(picks), self.width), self.pad, dtype=torch.long)
        att = torch.zeros((len(picks), self.width), dtype=torch.long)
        tgt = torch.zeros((len(picks), self.width - 1), dtype=torch.bool)
        for i, (seq, prompt_len) in enumerate(picks):
            ids[i, :len(seq)] = torch.tensor(seq)
            att[i, :len(seq)] = 1
            tgt[i, prompt_len - 1:len(seq) - 1] = True
        return ids.to(device), att.to(device), tgt.to(device)


class KELossTrainer(Trainer):
    """Three-term knowledge-editing objective.

        L = L_ce  +  lambda_forget * L_forget  +  lambda_retain * L_retain

    L_ce      the usual next-token loss on the Self-Study answers (unchanged).
    L_forget  a HINGE on the old answer's log-prob in the forget set: relu(logp - tau).
              Bounded on purpose -- the usual "maximise NLL" unlearning term is unbounded
              below and lets a few tokens dominate the gradient. The hinge stops pushing
              once the old answer is already unlikely (below tau).
    L_retain  KL to the UN-FINETUNED model on the retain set, so suppressing the old value
              for the edited subject does not drag it down for its neighbours.

    The reference model is these same weights with the LoRA adapter switched off
    (`disable_adapter`), so no second model and no second GPU is needed.
    """

    def __init__(self, *args, ke_sets=None, lambda_forget=1.0, lambda_retain=1.0, tau=None,
                 forget_batch=4, retain_batch=4, ke_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ke_sets = ke_sets
        self.lambda_forget = lambda_forget
        self.lambda_retain = lambda_retain
        # default tau = log(0.05): only penalise while the old answer still has more than
        # 5% probability at that position.
        self.tau = math.log(0.05) if tau is None else tau
        self.forget_batch = forget_batch
        self.retain_batch = retain_batch
        self.rng = random.Random(ke_seed)
        self._ke_logs = {}
        # The per-step term is one 4-sequence draw, so a single 0 says little; track how
        # often the hinge is active at all (it goes quiet once the old answer is below tau).
        self._forget_fired = 0
        self._forget_steps = 0

    def _answer_logprobs(self, model, ids, att, tgt):
        logprobs = model(input_ids=ids, attention_mask=att).logits[:, :-1].log_softmax(-1)
        gold = ids[:, 1:].unsqueeze(-1)
        return logprobs, logprobs.gather(-1, gold).squeeze(-1)[tgt]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce = outputs.loss
        device = ce.device
        forget = retain = ce.new_zeros(())

        if self.lambda_forget > 0 and self.ke_sets.forget:
            ids, att, tgt = self.ke_sets.batch("forget", self.forget_batch, self.rng, device)
            _, gold_lp = self._answer_logprobs(model, ids, att, tgt)
            forget = torch.relu(gold_lp - self.tau).mean()

        if self.lambda_retain > 0 and self.ke_sets.retain:
            ids, att, tgt = self.ke_sets.batch("retain", self.retain_batch, self.rng, device)
            cur, _ = self._answer_logprobs(model, ids, att, tgt)
            # `model` may be wrapped (torch.compile / accelerate) and disable_adapter lives
            # on the PeftModel underneath; run the reference pass through the SAME object so
            # the adapter-off state actually applies.
            peft_model = self.accelerator.unwrap_model(model)
            with torch.no_grad():
                with peft_model.disable_adapter():
                    ref = peft_model(input_ids=ids, attention_mask=att).logits[:, :-1].log_softmax(-1)
            retain = F.kl_div(cur[tgt], ref[tgt], log_target=True, reduction="batchmean")

        self._forget_steps += 1
        self._forget_fired += int(forget.detach().item() > 0)
        self._ke_logs = {"ce": ce.detach().item(), "forget": forget.detach().item(),
                         "retain": retain.detach().item(),
                         "forget_fired_pct": 100 * self._forget_fired / self._forget_steps}
        loss = ce + self.lambda_forget * forget + self.lambda_retain * retain
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Surface the three terms separately: if a run degrades you need to see WHICH term.
        if self._ke_logs:
            logs = {**logs, **{f"ke/{k}": v for k, v in self._ke_logs.items()}}
        return super().log(logs, *args, **kwargs)


def compile_model_if_needed(model, use_compile: bool = True):
    """Compile model with torch.compile if requested and available.
    
    Note: torch.compile may have compatibility issues with PEFT models.
    If compilation fails, training will continue without compilation.
    """
    if use_compile and hasattr(torch, 'compile'):
        print("Attempting to compile model with torch.compile...")
        print("Note: PEFT models may have compilation issues. If this fails, training will continue without compilation.")
        try:
            # Try to compile - this may fail with PEFT models
            compiled_model = torch.compile(model)
            print("Model compiled successfully")
            return compiled_model
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Continuing training without compilation (this is normal for PEFT models)")
            return model
    return model


def train(
    parquet_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 2048,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    save_steps: int = 500,
    logging_steps: int = 10,
    use_compile: bool = True,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    wandb_project: str = "cartridges-lora-finetune-anyedit-baseline",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    seed: int = None,
    ke_loss: bool = False,
    ke_data_file: str = None,
    lambda_forget: float = 1.0,
    lambda_retain: float = 1.0,
    tau: float = None,
    ke_forget_batch: int = 4,
    ke_retain_batch: int = 4,
    ke_max_edits: int = 0,
):
    """Main training function."""
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name or Path(output_dir).name,
        config={
            "parquet_path": parquet_path,
            "output_dir": output_dir,
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "max_length": max_length,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "use_compile": use_compile,
            "bf16": bf16,
            "gradient_checkpointing": gradient_checkpointing,
            "seed": seed,
            "ke_loss": ke_loss,
            "lambda_forget": lambda_forget if ke_loss else None,
            "lambda_retain": lambda_retain if ke_loss else None,
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --seed makes the LoRA init and the data order reproducible. Unset keeps the
    # historical behaviour: data order from the Trainer default (42), LoRA init unseeded.
    # The 90/10 train/val split stays at seed 42 either way, so a seed run trains on
    # exactly the same rows.
    if seed is not None:
        from transformers import set_seed
        set_seed(seed)
        print(f"Seed: {seed}")
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with more conservative memory settings
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        max_memory={0: "28GB"},  # Reserve some memory for operations
    )
    
    # Enable gradient checkpointing to save memory
    if gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Create LoRA model
    print("Creating LoRA model...")
    model = create_lora_model(model, lora_r, lora_alpha, lora_dropout)

    # Prepare dataset
    dataset = prepare_dataset(parquet_path, tokenizer, max_length)
    
    # Split into train/val (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Dynamic padding: pad each batch to its longest sequence (rounded to a
    # multiple of 8), padding labels with -100 so padded positions never
    # contribute to the loss. Replaces the previous fixed 2048 padding.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        torch_compile=use_compile,
        torch_compile_backend="inductor",
        label_names=["labels"],
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        bf16=bf16,
        bf16_full_eval=bf16,
        fp16=not bf16,  # Use fp16 if bf16 not available
        fp16_full_eval=not bf16,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,  # Reduced from 4 to save memory
        report_to="wandb",  # Enable wandb logging
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch_fused",  # Use fused optimizer for efficiency
        max_grad_norm=1.0,  # Gradient clipping
        **({"seed": seed} if seed is not None else {}),
    )
    
    # Create trainer. Without --ke-loss this is the plain Trainer, identical to every run
    # before the objective existed.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if ke_loss:
        ke_sets = KESets(ke_data_file, tokenizer, max_edits=ke_max_edits)
        print(f"KE loss ON: lambda_forget={lambda_forget} lambda_retain={lambda_retain} "
              f"tau={tau if tau is not None else 'log(0.05)'} "
              f"forget_batch={ke_forget_batch} retain_batch={ke_retain_batch}")
        trainer = KELossTrainer(**trainer_kwargs, ke_sets=ke_sets, lambda_forget=lambda_forget,
                                lambda_retain=lambda_retain, tau=tau,
                                forget_batch=ke_forget_batch, retain_batch=ke_retain_batch,
                                ke_seed=seed or 0)
    else:
        trainer = Trainer(**trainer_kwargs)
    
    # Train (autocast is handled by TrainingArguments bf16/fp16 flags)
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save config.json with all training parameters
    config_path = Path(output_dir) / "config.json"
    config_data = {
        "parquet_path": parquet_path,
        "output_dir": output_dir,
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "use_compile": use_compile,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_run_name": wandb_run_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Finish wandb run
    wandb.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Knowledge Editing")
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Path to the dataset.parquet file from self-study synthesis"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_output",
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (r)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,  # Reduced default from 4 to 2
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,  # Increased default from 8 to 16 to compensate
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile"
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 (use float32)"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cartridges-lora-finetune",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity/team name (optional)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (optional, defaults to output_dir name)"
    )
    parser.add_argument(
        "--ke-loss", action="store_true",
        help="Three-term objective: CE on the Self-Study answers + a bounded forget term on "
             "each edit's own question completed with the OLD answer + KL to the frozen base "
             "on the neighbour prompts. OFF by default (plain CE, as every run so far)."
    )
    parser.add_argument(
        "--ke-data-file", type=str, default=str(Path(__file__).resolve().parent / "samples" / "CounterFact.json"),
        help="CounterFact JSON the forget and retain sets are built from (--ke-loss only)"
    )
    parser.add_argument("--lambda-forget", type=float, default=1.0, help="weight of the forget term")
    parser.add_argument("--lambda-retain", type=float, default=1.0, help="weight of the KL retain term")
    parser.add_argument("--tau", type=float, default=None,
                        help="log-prob threshold of the forget hinge (default log(0.05))")
    parser.add_argument("--ke-forget-batch", type=int, default=4, help="forget sequences per step")
    parser.add_argument("--ke-retain-batch", type=int, default=4, help="retain sequences per step")
    parser.add_argument("--ke-max-edits", type=int, default=0,
                        help="build the KE sets from the first N edits only (canary runs)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for LoRA init and data order (default: unset = historical behaviour; the train/val split is always seed 42)"
    )
    
    args = parser.parse_args()
    
    train(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_compile=not args.no_compile,
        bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        ke_loss=args.ke_loss,
        ke_data_file=args.ke_data_file,
        lambda_forget=args.lambda_forget,
        lambda_retain=args.lambda_retain,
        tau=args.tau,
        ke_forget_batch=args.ke_forget_batch,
        ke_retain_batch=args.ke_retain_batch,
        ke_max_edits=args.ke_max_edits,
    )