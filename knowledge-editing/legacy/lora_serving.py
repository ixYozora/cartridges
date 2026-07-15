"""
Quick chat script for LoRA fine-tuned model.

Usage:
    python chat_lora.py --model-path ./lora_qwen3_4b
"""

import torch
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load config to get base model name
    import json
    from pathlib import Path
    config_path = Path(model_path) / "config.json"
    base_model_name = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            base_model_name = cfg.get("model_name") or cfg.get("base_model_name_or_path")

    # Try to detect if this is a PEFT/LoRA adapter or a merged model
    adapter_config = Path(model_path) / "adapter_config.json"
    is_peft = adapter_config.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_peft and base_model_name:
        print(f"Detected LoRA adapter. Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("Loading model directly (merged or non-PEFT)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    print("Model loaded!\n")
    return model, tokenizer


def chat(model, tokenizer, max_new_tokens: int = 512, temperature: float = 0.7):
    history = []
    print("=" * 50)
    print("Chat with your fine-tuned model. Type 'quit' to exit, 'reset' to clear history.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = []
            print("--- History cleared ---\n")
            continue

        history.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        history.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a LoRA fine-tuned model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model/adapter")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    chat(model, tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature)