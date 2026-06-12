#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def train(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train.py"),
        "--model", args.model,
        "--data", args.data,
    ]
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.lora_rank:
        cmd.extend(["--lora-rank", str(args.lora_rank)])
    subprocess.run(cmd)

def chat(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "chat.py"),
        "--model", args.model,
    ]
    if args.quant:
        cmd.extend(["--quant", args.quant])
    subprocess.run(cmd)

def evaluate(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "evaluate.py"),
        "--model", args.model,
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(prog="sphinx")
    sub = parser.add_subparsers(dest="command", required=True)
    
    train_parser = sub.add_parser("train")
    train_parser.add_argument("-model", required=True)
    train_parser.add_argument("-data", required=True)
    train_parser.add_argument("-epochs", type=int)
    train_parser.add_argument("-lora-rank", type=int)
    train_parser.set_defaults(func=train)
    
    chat_parser = sub.add_parser("chat")
    chat_parser.add_argument("-model", required=True)
    chat_parser.add_argument("-quant")
    chat_parser.set_defaults(func=chat)
    
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("-model", required=True)
    eval_parser.add_argument("-dataset")
    eval_parser.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
