import argparse
import json
import sys
import torch
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--quant", default="4bit")
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def load_dataset(path):
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def build_prompt(question):
    return f"|<|im_start|>user\n{question}|>\n<|<|im_start|>assistant\n"


def evaluate_self_awareness(model, tokenizer, item):
    question = item["question"]
    expected_tags = item.get("expected_tags", [])
    forbidden_tags = item.get("forbidden_tags", [])
    
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    response_lower = response.lower()
    
    score = 0
    checks = []
    
    for tag in expected_tags:
        if tag.lower() in response_lower:
            score += 1
            checks.append(f"has:{tag}")
        else:
            checks.append(f"missing:{tag}")
    
    for tag in forbidden_tags:
        if tag.lower() in response_lower:
            score -= 1
            checks.append(f"forbidden:{tag}")
        else:
            checks.append(f"ok:{tag}")
    
    return {
        "question": question,
        "response": response,
        "score": max(0, score),
        "max_score": len(expected_tags),
        "checks": checks,
    }


def main():
    global args
    args = parse_args()
    
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=(args.quant == "4bit"),
    )
    FastLanguageModel.for_inference(model)
    
    print(f"Loading dataset: {args.dataset}", file=sys.stderr)
    dataset = load_dataset(args.dataset)
    
    results = []
    total_score = 0
    total_max = 0
    
    for item in dataset:
        result = evaluate_self_awareness(model, tokenizer, item)
        results.append(result)
        total_score += result["score"]
        total_max += result["max_score"]
    
    overall = (total_score / total_max * 100) if total_max > 0 else 0
    
    output = {
        "overall_score": round(overall, 2),
        "total_questions": len(dataset),
        "total_score": total_score,
        "total_max": total_max,
        "results": results,
    }
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
