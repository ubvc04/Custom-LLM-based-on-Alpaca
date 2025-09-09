#!/usr/bin/env python3
"""
Simple Alpaca Dataset Downloader
Downloads the Alpaca dataset and prepares it for training
"""

import os
import json
import urllib.request
from pathlib import Path

def download_alpaca_dataset():
    """Download the Alpaca dataset from GitHub"""
    print("🔥 Downloading Alpaca Dataset...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Alpaca dataset URL
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    output_path = data_dir / "alpaca_raw.json"
    
    try:
        print(f"📥 Downloading from {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded to {output_path}")
        
        # Load and process the data
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Total samples: {len(data)}")
        
        # Process and format samples
        processed_samples = []
        for sample in data:
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            output = sample.get('output', '')
            
            # Skip samples with very short outputs
            if len(output) < 10:
                continue
            
            # Format prompt
            if input_text:
                prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            
            processed_samples.append({
                'prompt': prompt,
                'completion': output,
                'instruction': instruction,
                'input': input_text,
                'output': output
            })
        
        print(f"🔧 Processed {len(processed_samples)} high-quality samples")
        
        # Create train/validation/test splits
        import random
        random.shuffle(processed_samples)
        
        total_samples = len(processed_samples)
        train_size = int(total_samples * 0.85)
        val_size = int(total_samples * 0.10)
        
        train_data = processed_samples[:train_size]
        val_data = processed_samples[train_size:train_size + val_size]
        test_data = processed_samples[train_size + val_size:]
        
        # Save splits
        with open(data_dir / "alpaca_train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(data_dir / "alpaca_validation.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(data_dir / "alpaca_test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Train: {len(train_data)} samples")
        print(f"✅ Validation: {len(val_data)} samples") 
        print(f"✅ Test: {len(test_data)} samples")
        
        # Create dataset info
        dataset_info = {
            'dataset_name': 'Alpaca Dataset',
            'version': '1.0.0',
            'splits': {
                'train': len(train_data),
                'validation': len(val_data),
                'test': len(test_data)
            },
            'total_samples': len(processed_samples),
            'format': 'Alpaca conversation format'
        }
        
        with open(data_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print("🎉 Dataset preparation completed successfully!")
        print("🚀 Ready to train the world's best Alpaca model!")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    download_alpaca_dataset()
