import argparse
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--data", required=True, help="Path to dataset directory or file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--output", default="outputs")
    return parser.parse_args()


def load_self_awareness_data(data_path):
    if os.path.isfile(data_path):
        if data_path.endswith(".json"):
            import json
            with open(data_path) as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    return Dataset.from_list(raw)
                return Dataset.from_dict(raw)
        elif data_path.endswith(".jsonl"):
            return Dataset.from_json(data_path)
    elif os.path.isdir(data_path):
        files = [os.path.join(data_path, f) for f in os.listdir(data_path)
                 if f.endswith((".json", ".jsonl"))]
        if not files:
            raise ValueError(f"No .json or .jsonl files in {data_path}")
        datasets = []
        for f in files:
            if f.endswith(".jsonl"):
                datasets.append(Dataset.from_json(f))
            else:
                import json
                with open(f) as fh:
                    raw = json.load(fh)
                    if isinstance(raw, list):
                        datasets.append(Dataset.from_list(raw))
                    else:
                        datasets.append(Dataset.from_dict(raw))
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)
    raise ValueError(f"Cannot load data from {data_path}")


def format_dataset(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples.get("input", [""])[i] if "input" in examples else ""
        output = examples["output"][i]
        
        if input_text:
            prompt = f"|<|im_start|>user\n{instruction}\n{input_text}|>\n<|<|im_start|>assistant\n{output}|>"
        else:
            prompt = f"|<|im_start|>user\n{instruction}|>\n<|<|im_start|>assistant\n{output}|>"
        texts.append(prompt)
    return {"text": texts}


def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    print(f"Applying LoRA (r={args.lora_rank}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print(f"Loading dataset from: {args.data}")
    dataset = load_self_awareness_data(args.data)
    
    if "text" not in dataset.column_names:
        dataset = dataset.map(format_dataset, batched=True)
    
    print(f"Training samples: {len(dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            save_strategy="epoch",
        ),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
