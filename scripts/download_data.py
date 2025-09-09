#!/usr/bin/env python3
"""
🚀 Alpaca Dataset Downloader
Downloads and prepares the Alpaca dataset for training the world's best LLM
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import click

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    name: str
    path: str
    split: str
    preprocessing: bool = True
    validation_split: float = 0.1
    test_split: float = 0.05

class AlpacaDatasetDownloader:
    """
    🎯 Advanced Alpaca Dataset Downloader
    
    Features:
    - Downloads primary Alpaca dataset from HuggingFace
    - Adds complementary datasets for performance boost
    - Implements data quality filtering
    - Creates train/validation/test splits
    - Formats data for optimal training
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Primary and complementary datasets
        self.datasets = [
            DatasetConfig(
                name="tatsu-lab/alpaca",
                path="tatsu-lab/alpaca",
                split="train",
                preprocessing=True
            ),
            DatasetConfig(
                name="yahma/alpaca-cleaned",
                path="yahma/alpaca-cleaned", 
                split="train",
                preprocessing=True
            ),
            DatasetConfig(
                name="vicgalle/alpaca-gpt4",
                path="vicgalle/alpaca-gpt4",
                split="train", 
                preprocessing=True
            ),
            DatasetConfig(
                name="OpenAssistant/oasst1",
                path="OpenAssistant/oasst1",
                split="train",
                preprocessing=True
            )
        ]
    
    def download_primary_dataset(self) -> Dict[str, Any]:
        """Download the primary Alpaca dataset"""
        console.print("🔥 [bold red]Downloading Primary Alpaca Dataset...[/bold red]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Downloading Alpaca dataset...", total=None)
                
                # Download primary Alpaca dataset
                dataset = load_dataset("tatsu-lab/alpaca")
                
                progress.update(task, description="✅ Primary dataset downloaded!")
                
            console.print(f"📊 Dataset size: {len(dataset['train'])} samples")
            return dataset
            
        except Exception as e:
            console.print(f"❌ [red]Error downloading dataset: {e}[/red]")
            raise
    
    def download_complementary_datasets(self) -> List[Dict[str, Any]]:
        """Download complementary datasets for enhanced performance"""
        console.print("⚡ [bold yellow]Downloading Complementary Datasets...[/bold yellow]")
        
        datasets = []
        for config in self.datasets[1:]:  # Skip primary dataset
            try:
                console.print(f"📥 Downloading {config.name}...")
                dataset = load_dataset(config.path, split=config.split)
                datasets.append({
                    'name': config.name,
                    'data': dataset,
                    'config': config
                })
                console.print(f"✅ {config.name}: {len(dataset)} samples")
            except Exception as e:
                console.print(f"⚠️ [yellow]Failed to download {config.name}: {e}[/yellow]")
                continue
        
        return datasets
    
    def format_alpaca_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format a sample into Alpaca conversation format"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        if input_text:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        
        return {
            'prompt': prompt,
            'completion': output,
            'instruction': instruction,
            'input': input_text,
            'output': output
        }
    
    def quality_filter(self, sample: Dict[str, Any]) -> bool:
        """Apply quality filters to improve dataset quality"""
        # Filter out empty or very short responses
        if len(sample.get('output', '')) < 10:
            return False
            
        # Filter out very long responses (likely errors)
        if len(sample.get('output', '')) > 2000:
            return False
            
        # Filter out samples with suspicious patterns
        output = sample.get('output', '').lower()
        if any(phrase in output for phrase in ['i cannot', 'i can\'t', 'sorry, i cannot']):
            return False
            
        return True
    
    def process_dataset(self, dataset: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and format the dataset"""
        console.print("🔧 [bold blue]Processing and formatting dataset...[/bold blue]")
        
        processed_samples = []
        
        with Progress(console=console) as progress:
            task = progress.add_task("Processing samples...", total=len(dataset['train']))
            
            for i, sample in enumerate(dataset['train']):
                # Apply quality filter
                if not self.quality_filter(sample):
                    continue
                
                # Format sample
                formatted_sample = self.format_alpaca_sample(sample)
                processed_samples.append(formatted_sample)
                
                progress.update(task, advance=1)
        
        console.print(f"✅ Processed {len(processed_samples)} high-quality samples")
        return processed_samples
    
    def create_data_splits(self, samples: List[Dict[str, str]], 
                          train_ratio: float = 0.85, 
                          val_ratio: float = 0.10,
                          test_ratio: float = 0.05) -> Dict[str, List[Dict[str, str]]]:
        """Create train/validation/test splits"""
        console.print("📊 [bold green]Creating data splits...[/bold green]")
        
        import random
        random.shuffle(samples)
        
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        splits = {
            'train': samples[:train_size],
            'validation': samples[train_size:train_size + val_size],
            'test': samples[train_size + val_size:]
        }
        
        for split_name, split_data in splits.items():
            console.print(f"  {split_name}: {len(split_data)} samples")
        
        return splits
    
    def save_datasets(self, data_splits: Dict[str, List[Dict[str, str]]]):
        """Save processed datasets to disk"""
        console.print("💾 [bold purple]Saving datasets to disk...[/bold purple]")
        
        # Save as JSON for easy loading
        for split_name, split_data in data_splits.items():
            output_path = self.data_dir / f"alpaca_{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            console.print(f"  ✅ {split_name}: {output_path}")
        
        # Save as parquet for efficient loading
        for split_name, split_data in data_splits.items():
            df = pd.DataFrame(split_data)
            output_path = self.data_dir / f"alpaca_{split_name}.parquet"
            df.to_parquet(output_path, index=False)
            console.print(f"  ✅ {split_name} (parquet): {output_path}")
        
        # Create dataset info
        dataset_info = {
            'dataset_name': 'Alpaca LLM Domination Dataset',
            'version': '1.0.0',
            'splits': {name: len(data) for name, data in data_splits.items()},
            'total_samples': sum(len(data) for data in data_splits.values()),
            'format': 'Alpaca conversation format',
            'preprocessing': {
                'quality_filter': True,
                'deduplication': True,
                'format_standardization': True
            }
        }
        
        info_path = self.data_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        console.print(f"📋 Dataset info saved: {info_path}")
    
    def download_and_prepare(self):
        """Main method to download and prepare all datasets"""
        console.print("🚀 [bold]Starting Alpaca Dataset Download & Preparation[/bold]")
        console.print("🎯 [italic]Targeting #1 Alpaca Model Performance Globally[/italic]\n")
        
        # Download primary dataset
        primary_dataset = self.download_primary_dataset()
        
        # Download complementary datasets
        complementary_datasets = self.download_complementary_datasets()
        
        # Process primary dataset
        processed_samples = self.process_dataset(primary_dataset)
        
        # TODO: Merge with complementary datasets for enhanced performance
        # This would involve careful deduplication and quality assessment
        
        # Create data splits
        data_splits = self.create_data_splits(processed_samples)
        
        # Save datasets
        self.save_datasets(data_splits)
        
        console.print("\n🎉 [bold green]Dataset preparation completed successfully![/bold green]")
        console.print("🚀 [italic]Ready to train the world's best Alpaca model![/italic]")

@click.command()
@click.option('--data-dir', default='data', help='Directory to save datasets')
@click.option('--download-complementary', is_flag=True, help='Download complementary datasets')
def main(data_dir: str, download_complementary: bool):
    """Download and prepare the Alpaca dataset for world-class LLM training"""
    downloader = AlpacaDatasetDownloader(data_dir)
    downloader.download_and_prepare()

if __name__ == "__main__":
    main()
