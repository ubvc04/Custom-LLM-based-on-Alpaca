#!/usr/bin/env python3
"""
⚡ ULTRA-FAST Alpaca Training - Complete in 2-3 minutes!
Uses the most efficient techniques for lightning-fast model training
"""

import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

print("⚡ ULTRA-FAST ALPACA TRAINING ⚡")
print("🎯 Target: Complete in under 3 minutes!")
print("=" * 50)

class UltraFastDataset(Dataset):
    """Ultra-minimal dataset for lightning training"""
    
    def __init__(self, data_path: str, tokenizer, max_samples: int = 100, max_length: int = 128):
        print(f"📊 Loading ultra-fast dataset...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use only the first N samples for speed
        self.data = data[:max_samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"✅ Using {len(self.data)} samples (ultra-fast mode)")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Create a simple prompt-response format
        text = f"Question: {sample['instruction']}\nAnswer: {sample['output']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def ultra_fast_train():
    """Ultra-fast training function"""
    
    # Use the smallest possible model for speed
    model_name = "gpt2"  # Smallest model (124M parameters)
    
    print(f"🚀 Loading ultra-light model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"📱 Device: {device}")
    print(f"💾 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create ultra-fast dataset
    data_path = Path("data/alpaca_train.json")
    if not data_path.exists():
        print("❌ Training data not found! Please run setup first.")
        return
    
    dataset = UltraFastDataset(str(data_path), tokenizer, max_samples=100, max_length=128)
    
    # Ultra-fast dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=8,  # Moderate batch size for speed
        shuffle=True,
        num_workers=0,  # No multiprocessing for simplicity
        pin_memory=False
    )
    
    # Setup optimizer with high learning rate for fast convergence
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # Minimal scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=len(dataloader)
    )
    
    # Training loop
    model.train()
    
    print("\n🔥 Starting ultra-fast training...")
    print(f"📊 Batches to process: {len(dataloader)}")
    
    start_time = time.time()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", ncols=100)
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    training_time = time.time() - start_time
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Training time: {training_time:.1f} seconds")
    print(f"📊 Final average loss: {total_loss / len(dataloader):.4f}")
    
    # Save the model
    output_dir = Path("ultra_fast_model")
    output_dir.mkdir(exist_ok=True)
    
    print(f"💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Model saved successfully!")
    
    # Quick test
    print("\n🧪 Quick model test:")
    test_prompt = "Question: What is artificial intelligence?\nAnswer:"
    
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Model response: {response}")
    
    print(f"\n🏆 SUCCESS! Ultra-fast training completed in {training_time:.1f} seconds!")
    print("🚀 Your lightning-fast Alpaca model is ready!")

if __name__ == "__main__":
    try:
        ultra_fast_train()
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")
