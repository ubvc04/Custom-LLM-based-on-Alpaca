#!/usr/bin/env python3
"""
LIGHTNING-FAST Alpaca Training (Under 5 minutes!)
Using the most aggressive optimizations for maximum speed
"""

import json
import time
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from rich.console import Console

console = Console()

class LightningDataset(Dataset):
    """Minimal dataset for maximum speed"""
    
    def __init__(self, data_path: str, tokenizer, max_samples: int = 1000):
        console.print(f"[blue]Loading lightning dataset...[/blue]")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use tiny subset for speed
        self.data = data[:max_samples]
        self.tokenizer = tokenizer
        
        console.print(f"[green]Using {len(self.data)} samples for lightning training[/green]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Very simple formatting
        text = f"Q: {sample['instruction']}\nA: {sample['output']}<|endoftext|>"
        
        # Quick tokenization
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=256,  # Very short for speed
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def lightning_train():
    """Lightning-fast training in under 5 minutes"""
    
    console.print("[bold red]⚡ LIGHTNING-FAST ALPACA TRAINING ⚡[/bold red]")
    console.print("[yellow]Target: Complete in under 5 minutes![/yellow]\n")
    
    start_time = time.time()
    
    # Use the smallest possible model for speed
    model_name = "gpt2"  # Very fast and small
    console.print(f"[blue]Loading model: {model_name}[/blue]")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Check data exists
    data_path = Path("data/alpaca_train.json")
    if not data_path.exists():
        console.print("[red]No training data found! Run setup first.[/red]")
        return
    
    # Create lightning dataset
    train_dataset = LightningDataset(str(data_path), tokenizer, max_samples=500)
    
    # Lightning-fast training arguments
    training_args = TrainingArguments(
        output_dir="experiments/lightning-alpaca",
        overwrite_output_dir=True,
        
        # Minimal training for maximum speed
        num_train_epochs=1,
        per_device_train_batch_size=16,  # Large batch
        gradient_accumulation_steps=1,   # No accumulation
        
        # Aggressive learning
        learning_rate=1e-3,  # High learning rate
        warmup_steps=10,     # Minimal warmup
        
        # Speed optimizations
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        
        # Minimal logging/saving
        logging_steps=50,
        save_steps=1000,
        eval_strategy="no",  # Fixed parameter name
        report_to=[],
        
        # No checkpointing for speed
        save_total_limit=1,
        save_safetensors=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    console.print("[bold green]Starting lightning training...[/bold green]")
    train_start = time.time()
    
    # Train!
    trainer.train()
    
    train_time = time.time() - train_start
    total_time = time.time() - start_time
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Results
    console.print(f"\n[bold green]⚡ LIGHTNING TRAINING COMPLETED! ⚡[/bold green]")
    console.print(f"[cyan]Training time: {train_time:.2f} seconds[/cyan]")
    console.print(f"[cyan]Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)[/cyan]")
    
    if total_time < 300:  # 5 minutes
        console.print("[bold green]🎉 SUCCESS: Completed in under 5 minutes![/bold green]")
    elif total_time < 600:  # 10 minutes
        console.print("[yellow]🎉 SUCCESS: Completed in under 10 minutes![/yellow]")
    else:
        console.print(f"[red]Training took {total_time/60:.2f} minutes[/red]")
    
    # Quick test
    console.print("\n[blue]Testing the lightning model...[/blue]")
    test_lightning_model(model, tokenizer)
    
    console.print(f"\n[green]Model saved to: {training_args.output_dir}[/green]")

def test_lightning_model(model, tokenizer):
    """Quick test of the lightning model"""
    
    test_prompt = "Q: What is machine learning?\nA:"
    
    inputs = tokenizer.encode(test_prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(test_prompt):].strip()
    
    console.print(f"[cyan]Test prompt: What is machine learning?[/cyan]")
    console.print(f"[green]Model response: {answer}[/green]")

if __name__ == "__main__":
    lightning_train()
