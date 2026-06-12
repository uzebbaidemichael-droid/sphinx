import argparse
import sys
import torch
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quant", default="4bit")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def build_prompt(messages):
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"|<|im_start|>system\n{content}|>")
        elif role == "user":
            parts.append(f"|<|im_start|>user\n{content}|>")
        elif role == "assistant":
            parts.append(f"|<|im_start|>assistant\n{content}|>")
    parts.append("|<|im_start|>assistant\n")
    return "".join(parts)


def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=(args.quant == "4bit"),
    )
    FastLanguageModel.for_inference(model)
    
    print("Ready", flush=True)
    
    while True:
        try:
            line = input()
        except EOFError:
            break
        
        if not line.strip():
            continue
        
        import json
        try:
            messages = json.loads(line)
        except json.JSONDecodeError:
            print("Error: invalid JSON", flush=True)
            continue
        
        prompt = build_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(response, flush=True)


if __name__ == "__main__":
    main()
