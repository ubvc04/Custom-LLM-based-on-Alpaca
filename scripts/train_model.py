#!/usr/bin/env python3
"""
🚀 Advanced Training Pipeline for Alpaca Domination
World-class training techniques for superior model performance
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import transformers
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

import wandb
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.alpaca_domination import (
    AlpacaDominationForCausalLM, 
    get_alpaca_7b_config,
    get_alpaca_13b_config, 
    get_alpaca_30b_config,
    ModelConfig
)

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class TrainingConfig:
    """🎯 Advanced training configuration for world-class performance"""
    
    # Model configuration
    model_size: str = "7b"  # 7b, 13b, 30b
    model_name: str = "alpaca-domination-7b"
    base_model: str = "meta-llama/Llama-2-7b-hf"
    
    # Data configuration
    data_dir: str = "data"
    max_seq_length: int = 2048
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 32
    
    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Advanced optimizations
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_deepspeed: bool = True
    use_lora: bool = False
    use_qlora: bool = False
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Constitutional AI
    use_constitutional_loss: bool = True
    constitutional_weight: float = 0.1
    
    # Monitoring and logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    use_wandb: bool = True
    wandb_project: str = "alpaca-domination"
    
    # Output directories
    output_dir: str = "experiments/alpaca-domination-7b"
    logging_dir: str = "logs"
    cache_dir: str = ".cache"
    
    # Hardware optimization
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Evaluation
    eval_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Advanced features
    use_cpu_offload: bool = False
    use_8bit_adam: bool = True
    group_by_length: bool = True

class AlpacaDataset(Dataset):
    """🔥 Optimized Alpaca Dataset for high-performance training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            self.data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        console.print(f"📊 Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Format prompt for training
        prompt = sample['prompt']
        completion = sample['completion']
        
        # Tokenize prompt and completion
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine and truncate
        input_ids = prompt_tokens + completion_tokens + [self.tokenizer.eos_token_id]
        input_ids = input_ids[:self.max_length]
        
        # Create labels (mask prompt tokens for loss calculation)
        labels = [-100] * len(prompt_tokens) + completion_tokens + [self.tokenizer.eos_token_id]
        labels = labels[:self.max_length]
        
        # Pad sequences
        attention_mask = [1] * len(input_ids)
        
        # Pad to max length
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

class ConstitutionalTrainer(Trainer):
    """🎯 Advanced Trainer with Constitutional AI integration"""
    
    def __init__(self, constitutional_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.constitutional_weight = constitutional_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with constitutional AI component"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Standard language modeling loss
        loss = outputs.get('loss')
        
        # Add constitutional loss if available
        constitutional_loss = outputs.get('constitutional_loss')
        if constitutional_loss is not None:
            total_loss = loss + self.constitutional_weight * constitutional_loss
            
            # Log constitutional loss
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    'constitutional_loss': constitutional_loss.item(),
                    'base_loss': loss.item(),
                    'total_loss': total_loss.item()
                })
        else:
            total_loss = loss
        
        return (total_loss, outputs) if return_outputs else total_loss

class AlpacaDominationTrainer:
    """
    🏆 Master Trainer for Alpaca Domination
    
    Implements state-of-the-art training techniques:
    - Mixed precision training
    - Gradient checkpointing
    - DeepSpeed optimization
    - Constitutional AI integration
    - Advanced monitoring and logging
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_environment()
        self.setup_model_and_tokenizer()
        self.setup_datasets()
        self.setup_training()
    
    def setup_environment(self):
        """Setup training environment and monitoring"""
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config.__dict__
            )
        
        # Setup distributed training if available
        if torch.cuda.device_count() > 1:
            console.print(f"🚀 Using {torch.cuda.device_count()} GPUs for training")
            
        console.print("🎯 [bold]Training Environment Setup Complete[/bold]")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        console.print("🧠 [bold blue]Setting up model and tokenizer...[/bold blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model configuration
        if self.config.model_size == "7b":
            model_config = get_alpaca_7b_config()
        elif self.config.model_size == "13b":
            model_config = get_alpaca_13b_config()
        elif self.config.model_size == "30b":
            model_config = get_alpaca_30b_config()
        else:
            raise ValueError(f"Unsupported model size: {self.config.model_size}")
        
        # Update config with training settings
        model_config.use_flash_attention = self.config.use_flash_attention
        model_config.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        model_config.use_constitutional_loss = self.config.use_constitutional_loss
        
        # Initialize model
        self.model = AlpacaDominationForCausalLM(model_config)
        
        # Load from base model if available
        try:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            )
            
            # Copy compatible weights
            self._copy_compatible_weights(base_model, self.model)
            console.print("✅ Successfully loaded base model weights")
            
        except Exception as e:
            console.print(f"⚠️ Could not load base model: {e}")
            console.print("🔄 Training from scratch")
        
        # Setup LoRA if enabled
        if self.config.use_lora or self.config.use_qlora:
            self.setup_lora()
        
        console.print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        console.print(f"🔥 Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _copy_compatible_weights(self, source_model, target_model):
        """Copy compatible weights from source to target model"""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        copied_layers = 0
        for name, param in target_dict.items():
            if name in source_dict and source_dict[name].shape == param.shape:
                param.data.copy_(source_dict[name].data)
                copied_layers += 1
        
        console.print(f"✅ Copied {copied_layers} compatible layers")
    
    def setup_lora(self):
        """Setup LoRA/QLoRA for efficient fine-tuning"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
            
            if self.config.use_qlora:
                # Prepare model for 4-bit training
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            console.print("✅ LoRA/QLoRA setup complete")
            
        except ImportError:
            console.print("⚠️ PEFT not available. Install with: pip install peft")
            self.config.use_lora = False
            self.config.use_qlora = False
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        console.print("📊 [bold green]Setting up datasets...[/bold green]")
        
        data_dir = Path(self.config.data_dir)
        
        # Load datasets
        train_path = data_dir / "alpaca_train.json"
        val_path = data_dir / "alpaca_validation.json"
        
        if not train_path.exists():
            console.print(f"❌ Training data not found at {train_path}")
            console.print("🔄 Please run download_data.py first")
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        self.train_dataset = AlpacaDataset(
            str(train_path), 
            self.tokenizer, 
            max_length=self.config.max_seq_length
        )
        
        if val_path.exists():
            self.eval_dataset = AlpacaDataset(
                str(val_path), 
                self.tokenizer, 
                max_length=self.config.max_seq_length
            )
        else:
            console.print("⚠️ Validation data not found, using train split")
            self.eval_dataset = None
        
        console.print(f"✅ Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            console.print(f"✅ Validation samples: {len(self.eval_dataset)}")
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        console.print("⚙️ [bold yellow]Setting up training configuration...[/bold yellow]")
        
        # Calculate total steps
        total_steps = (len(self.train_dataset) // 
                      (self.config.train_batch_size * self.config.gradient_accumulation_steps)) * self.config.num_epochs
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            
            # Precision and performance
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            group_by_length=self.config.group_by_length,
            
            # Logging and evaluation
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            evaluation_strategy=self.config.eval_strategy if self.eval_dataset else "no",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            load_best_model_at_end=self.config.load_best_model_at_end,
            
            # Monitoring
            report_to=["wandb"] if self.config.use_wandb else [],
            run_name=f"{self.config.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # DeepSpeed
            deepspeed="config/deepspeed_z3.json" if self.config.use_deepspeed else None,
            
            # Additional optimizations
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Initialize trainer
        self.trainer = ConstitutionalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            constitutional_weight=self.config.constitutional_weight,
        )
        
        console.print(f"🎯 Total training steps: {total_steps:,}")
        console.print("✅ Training setup complete")
    
    def train(self):
        """Execute the training process"""
        console.print("🚀 [bold red]Starting Alpaca Domination Training![/bold red]")
        console.print("🎯 [italic]Targeting #1 Global Performance[/italic]\n")
        
        # Display training summary
        self.display_training_summary()
        
        # Start training
        start_time = time.time()
        
        try:
            # Resume from checkpoint if available
            checkpoint = None
            if (Path(self.config.output_dir) / "pytorch_model.bin").exists():
                checkpoint = self.config.output_dir
                console.print(f"🔄 Resuming from checkpoint: {checkpoint}")
            
            # Train the model
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            
            # Training completed
            training_time = time.time() - start_time
            
            console.print(f"\n🎉 [bold green]Training completed successfully![/bold green]")
            console.print(f"⏱️  Training time: {training_time:.2f} seconds")
            console.print(f"📊 Final train loss: {train_result.training_loss:.4f}")
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Run final evaluation
            if self.eval_dataset:
                eval_results = self.trainer.evaluate()
                console.print(f"📊 Final eval loss: {eval_results['eval_loss']:.4f}")
            
            console.print(f"💾 Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            console.print(f"❌ [red]Training failed: {e}[/red]")
            raise
        
        finally:
            if self.config.use_wandb:
                wandb.finish()
    
    def display_training_summary(self):
        """Display comprehensive training summary"""
        table = Table(title="🏆 Alpaca Domination Training Configuration")
        
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Model Size", self.config.model_size)
        table.add_row("Model Name", self.config.model_name)
        table.add_row("Base Model", self.config.base_model)
        table.add_row("Training Samples", f"{len(self.train_dataset):,}")
        table.add_row("Validation Samples", f"{len(self.eval_dataset):,}" if self.eval_dataset else "None")
        table.add_row("Max Sequence Length", f"{self.config.max_seq_length:,}")
        table.add_row("Batch Size", str(self.config.train_batch_size))
        table.add_row("Gradient Accumulation", str(self.config.gradient_accumulation_steps))
        table.add_row("Learning Rate", f"{self.config.learning_rate:.2e}")
        table.add_row("Epochs", str(self.config.num_epochs))
        table.add_row("Mixed Precision", "BF16" if self.config.bf16 else "FP16" if self.config.fp16 else "FP32")
        table.add_row("Flash Attention", "✅" if self.config.use_flash_attention else "❌")
        table.add_row("Constitutional AI", "✅" if self.config.use_constitutional_loss else "❌")
        table.add_row("LoRA/QLoRA", "✅" if self.config.use_lora or self.config.use_qlora else "❌")
        table.add_row("DeepSpeed", "✅" if self.config.use_deepspeed else "❌")
        
        console.print(table)

def main():
    """Main training function"""
    # Training configuration
    config = TrainingConfig(
        model_size="7b",
        model_name="alpaca-domination-7b-v1",
        num_epochs=3,
        train_batch_size=4,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        use_flash_attention=True,
        use_constitutional_loss=True,
        use_wandb=True,
    )
    
    # Initialize trainer
    trainer = AlpacaDominationTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
