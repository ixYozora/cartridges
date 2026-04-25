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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
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


def format_conversation_for_training(conversation: Conversation, tokenizer) -> str:
    """Format a conversation into a training text using the chat template."""
    # Convert messages to dict format
    messages = [msg.to_message_dict() for msg in conversation.messages]
    
    # Apply chat template using tokenizer
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # Fallback: simple formatting if template fails
            print(f"Warning: Chat template failed: {e}. Using simple formatting.")
            formatted = ""
            for msg in messages:
                formatted += f"{msg['role']}: {msg['content']}\n"
    else:
        # Fallback: simple formatting
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
    
    return formatted


def prepare_dataset(parquet_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Prepare dataset from parquet file for training."""
    print(f"Loading conversations from {parquet_path}...")
    conversations = load_conversations_from_parquet(parquet_path)
    print(f"Loaded {len(conversations)} conversations")
    
    # Format conversations
    texts = []
    for conv in conversations:
        formatted = format_conversation_for_training(conv, tokenizer)
        texts.append(formatted)
    
    # Tokenize
    def tokenize_function(examples):
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


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
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
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
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
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
    )