import os
import math
import argparse
from datetime import datetime
from dataclasses import asdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer
from transformer_lm import TransformerModel, Config
from text_utils import read_and_clean_text_files


class TokenSequenceDataset(Dataset):
    """
    Dataset for causal language modelling
    """
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        
        if len(self.token_ids) <= block_size:
            raise ValueError(f"Need more than {block_size} tokens, got {len(self.token_ids)}")
    
    def __len__(self):
        return len(self.token_ids) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


def tokenize_text(tokenizer: Tokenizer, text: str) -> list[int]:
    _, ids = tokenizer.tokenize(text)
    return ids


def split_train_dev(token_ids: list[int], dev_fraction: float):
    split_idx = int(len(token_ids) * (1.0 - dev_fraction))
    train_ids = token_ids[:split_idx]
    dev_ids = token_ids[split_idx:]
    return train_ids, dev_ids


def evaluate(model, dev_loader, criterion, device, vocab_size):
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in dev_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    
    if total_batches == 0:
        return float("nan")
    
    return total_loss / total_batches


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    epoch,
    iteration,
    tokenizer_path,
    text_paths
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
        "config": asdict(config),
        "tokenizer_path": tokenizer_path,
        "text_paths": text_paths,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer language model from raw text files.")
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--tokenizer", type=str, default="data/hp_tokenizer.json")
    parser.add_argument("--checkpoint", type=str, default="data/hp_checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vector-dim", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    tokenizer = Tokenizer.load(args.tokenizer)
    
    print("Reading text files...")
    raw_text = read_and_clean_text_files(args.texts)
    
    print("Tokenizing text...")
    token_ids = tokenize_text(tokenizer, raw_text)
    
    print(f"Total tokens: {len(token_ids)}")
    
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = Config(**checkpoint["config"])
        
        model = TransformerModel(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        start_epoch = checkpoint["epoch"]
        global_iteration = checkpoint["iteration"]
        
    else:
        config = Config(
            vocab_size=tokenizer.vocab_size,
            number_of_transformer_blocks=args.blocks,
            number_of_attention_heads=args.heads,
            vector_dim=args.vector_dim,
            block_size=args.block_size,
            dropout_prob=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            no_of_epochs=args.epochs
        )
        
        model = TransformerModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        start_epoch = 0
        global_iteration = 0
        
    train_ids, dev_ids = split_train_dev(token_ids, args.dev_fraction)
    
    print(f"Train tkons: {len(train_ids)}")
    print(f"Dev tokens:   {len(dev_ids)}")
    train_dataset = TokenSequenceDataset(train_ids, config.block_size)
    dev_dataset = TokenSequenceDataset(dev_ids, config.block_size)
    pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    criterion = nn.CrossEntropyLoss()
    
    print(model)
    print(datetime.now().strftime("%X"), "Training starts")
    
    model.train()
    
    for epoch in range(start_epoch, config.no_of_epochs):
        running_loss = 0.0
        running_steps = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1)
            )
            loss.backward()
            
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            running_loss += loss.item()
            running_steps += 1
            global_iteration += 1
            
            if global_iteration % args.log_every == 0:
                train_loss = running_loss / max(1, running_steps)
                running_loss = 0.0
                running_steps = 0
                
                dev_loss = evaluate(model, dev_loader, criterion, device, config.vocab_size)
                
                if dev_loss < 20:
                    ppl = math.exp(dev_loss)
                    ppl_text = f", dev ppl={ppl:.2f}"
                else:
                    ppl_text = ""
                
                print(
                    f"{datetime.now().strftime('%X')} "
                    f"Epoch {epoch + 1}/{config.no_of_epochs}, "
                    f"iteration {global_iteration}, "
                    f"train loss={train_loss:.4f}, "
                    f"dev loss={dev_loss:.4f}" 
                    f"{ppl_text}"
                )
            
            if global_iteration % args.save_every == 0:
                save_checkpoint(args.checkpoint, model, optimizer, config,
                    epoch, global_iteration, args.tokenizer, args.texts)
        
        save_checkpoint(args.checkpoint, model, optimizer, config, epoch + 1,
            global_iteration, args.tokenizer, args.texts)
    
    print(datetime.now().strftime("%X"), "Training finished")


if __name__ == "__main__":
    train(parse_args())
    
