#!/usr/bin/env python3
"""
Ultra-Fast Training Script for Alpaca Domination
Training the world's best model in under 10 minutes!
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tqdm import tqdm

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class UltraFastConfig:
    """Ultra-fast training configuration for 10-minute training"""
    
    # Model configuration - Using smaller, efficient model
    model_name: str = "microsoft/DialoGPT-small"  # Fast base model
    model_size: str = "small"
    max_seq_length: int = 512  # Reduced for speed
    
    # Ultra-fast training parameters
    num_epochs: int = 1  # Single epoch for speed
    train_batch_size: int = 8  # Larger batch for efficiency
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4  # Reduced for speed
    
    # Aggressive learning rate for fast convergence
    learning_rate: float = 5e-4  # Higher learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # Reduced warmup
    max_grad_norm: float = 1.0
    
    # Speed optimizations
    use_gradient_checkpointing: bool = False  # Disabled for speed
    use_flash_attention: bool = False  # Keep simple for compatibility
    fp16: bool = True  # Mixed precision for speed
    bf16: bool = False
    dataloader_num_workers: int = 0  # No multiprocessing overhead
    
    # Reduced logging and saving for speed
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 1
    
    # Directories
    output_dir: str = "experiments/alpaca-fast"
    data_dir: str = "data"
    
    # Fast dataset processing
    max_train_samples: int = 5000  # Use subset for speed
    max_eval_samples: int = 500

class FastAlpacaDataset(Dataset):
    """Ultra-fast dataset for quick training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data efficiently
        console.print(f"[blue]Loading data from {data_path}...[/blue]")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Use subset for speed
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
            console.print(f"[yellow]Using {len(data)} samples for fast training[/yellow]")
        
        self.data = data
        console.print(f"[green]Loaded {len(self.data)} samples[/green]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Simple format for speed
        text = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}<|endoftext|>"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels same as input_ids for causal LM
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class UltraFastTrainer:
    """Ultra-fast trainer optimized for 10-minute training"""
    
    def __init__(self, config: UltraFastConfig):
        self.config = config
        self.setup_environment()
        self.setup_model_and_tokenizer()
        self.setup_datasets()
        self.setup_training()
    
    def setup_environment(self):
        """Setup training environment"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.device = "cuda"
            console.print(f"[green]Using GPU: {torch.cuda.get_device_name()}[/green]")
        else:
            self.device = "cpu"
            console.print("[yellow]Using CPU (training will be slower)[/yellow]")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for fast training"""
        console.print("[blue]Setting up model and tokenizer...[/blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            trust_remote_code=True
        )
        
        # Resize embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        console.print(f"[green]Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters[/green]")
    
    def setup_datasets(self):
        """Setup datasets for fast training"""
        console.print("[blue]Setting up datasets...[/blue]")
        
        data_dir = Path(self.config.data_dir)
        
        # Load training data
        train_path = data_dir / "alpaca_train.json"
        val_path = data_dir / "alpaca_validation.json"
        
        if not train_path.exists():
            console.print("[red]Training data not found! Run the setup first.[/red]")
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        self.train_dataset = FastAlpacaDataset(
            str(train_path), 
            self.tokenizer, 
            max_length=self.config.max_seq_length,
            max_samples=self.config.max_train_samples
        )
        
        if val_path.exists():
            self.eval_dataset = FastAlpacaDataset(
                str(val_path), 
                self.tokenizer, 
                max_length=self.config.max_seq_length,
                max_samples=self.config.max_eval_samples
            )
        else:
            self.eval_dataset = None
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        console.print("[blue]Setting up training configuration...[/blue]")
        
        # Training arguments optimized for speed
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimization for speed
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            
            # Precision and performance
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # Minimal logging for speed
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Disable unnecessary features for speed
            report_to=[],  # No wandb for speed
            remove_unused_columns=False,
            label_names=["labels"],
            
            # Speed optimizations
            ddp_find_unused_parameters=False,
            save_safetensors=False,  # Faster saving
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        console.print("[green]Training setup complete![/green]")
    
    def train(self):
        """Execute ultra-fast training"""
        console.print("[bold red]Starting Ultra-Fast Alpaca Training![/bold red]")
        console.print("[italic]Target: Complete in under 10 minutes[/italic]\n")
        
        # Display training summary
        self.display_training_summary()
        
        # Start training
        start_time = time.time()
        
        try:
            console.print("[bold green]Training started...[/bold green]")
            
            # Train the model
            train_result = self.trainer.train()
            
            # Training completed
            training_time = time.time() - start_time
            minutes = training_time / 60
            
            console.print(f"\n[bold green]Training completed successfully![/bold green]")
            console.print(f"[cyan]Training time: {training_time:.2f} seconds ({minutes:.2f} minutes)[/cyan]")
            console.print(f"[yellow]Final train loss: {train_result.training_loss:.4f}[/yellow]")
            
            # Check if under 10 minutes
            if minutes <= 10:
                console.print("[bold green]🎉 SUCCESS: Training completed in under 10 minutes![/bold green]")
            else:
                console.print(f"[yellow]Training took {minutes:.2f} minutes (target was 10 minutes)[/yellow]")
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Run final evaluation if available
            if self.eval_dataset:
                console.print("[blue]Running final evaluation...[/blue]")
                eval_results = self.trainer.evaluate()
                console.print(f"[cyan]Final eval loss: {eval_results['eval_loss']:.4f}[/cyan]")
            
            console.print(f"[green]Model saved to: {self.config.output_dir}[/green]")
            
            # Test the model
            self.test_model()
            
        except Exception as e:
            console.print(f"[red]Training failed: {e}[/red]")
            raise
    
    def test_model(self):
        """Quick test of the trained model"""
        console.print("[bold blue]Testing the trained model...[/bold blue]")
        
        # Test prompts
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?",
        ]
        
        for prompt in test_prompts:
            console.print(f"\n[cyan]Prompt: {prompt}[/cyan]")
            
            # Format as instruction
            formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors='pt')
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = response[len(formatted_prompt):].strip()
            
            console.print(f"[green]Response: {response_only}[/green]")
    
    def display_training_summary(self):
        """Display training summary"""
        table = Table(title="⚡ Ultra-Fast Training Configuration")
        
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Base Model", self.config.model_name)
        table.add_row("Training Samples", f"{len(self.train_dataset):,}")
        table.add_row("Validation Samples", f"{len(self.eval_dataset):,}" if self.eval_dataset else "None")
        table.add_row("Max Sequence Length", f"{self.config.max_seq_length:,}")
        table.add_row("Batch Size", str(self.config.train_batch_size))
        table.add_row("Gradient Accumulation", str(self.config.gradient_accumulation_steps))
        table.add_row("Learning Rate", f"{self.config.learning_rate:.2e}")
        table.add_row("Epochs", str(self.config.num_epochs))
        table.add_row("Mixed Precision", "FP16" if self.config.fp16 else "FP32")
        table.add_row("Target Time", "< 10 minutes")
        
        console.print(table)

def main():
    """Main ultra-fast training function"""
    console.print("[bold]🏆 ULTRA-FAST ALPACA TRAINING[/bold]")
    console.print("[italic]Training the world's best model in under 10 minutes![/italic]\n")
    
    # Ultra-fast configuration
    config = UltraFastConfig()
    
    # Initialize trainer
    trainer = UltraFastTrainer(config)
    
    # Start training
    trainer.train()
    
    console.print("\n[bold green]🎉 ULTRA-FAST TRAINING COMPLETED![/bold green]")
    console.print("[cyan]Your Alpaca model is ready for inference![/cyan]")

if __name__ == "__main__":
    main()
