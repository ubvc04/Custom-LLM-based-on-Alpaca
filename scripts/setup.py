#!/usr/bin/env python3
"""
🚀 Complete Setup Script for Alpaca Domination Project
Automated setup for the world's best LLM training environment
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import urllib.request

def run_command(command, description, check=True):
    """Run a command with error handling"""
    print(f"🔄 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def download_dataset():
    """Download the Alpaca dataset"""
    print("📊 Downloading Alpaca dataset...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        output_path = data_dir / "alpaca_raw.json"
        
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded dataset to {output_path}")
        
        # Process the data
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Total samples: {len(data)}")
        
        # Create formatted samples
        processed_samples = []
        for sample in data:
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            output = sample.get('output', '')
            
            if len(output) < 10:
                continue
            
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
        
        # Create splits
        import random
        random.shuffle(processed_samples)
        
        total = len(processed_samples)
        train_size = int(total * 0.85)
        val_size = int(total * 0.10)
        
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
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return False

def setup_python_environment():
    """Setup Python virtual environment and install dependencies"""
    print("🐍 Setting up Python environment...")
    
    # Create virtual environment if it doesn't exist
    if not Path("alpaca_llm_env").exists():
        if not run_command("python -m venv alpaca_llm_env", "Creating virtual environment"):
            return False
    
    # Determine the correct python executable path
    if os.name == 'nt':  # Windows
        python_exe = "alpaca_llm_env\\Scripts\\python.exe"
        pip_exe = "alpaca_llm_env\\Scripts\\pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = "alpaca_llm_env/bin/python"
        pip_exe = "alpaca_llm_env/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_exe} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install core dependencies
    core_packages = [
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "huggingface-hub>=0.19.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "rich>=13.7.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0"
    ]
    
    for package in core_packages:
        if not run_command(f"{pip_exe} install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    print("✅ Python environment setup completed")
    return True

def setup_frontend():
    """Setup React frontend"""
    print("🎨 Setting up frontend...")
    
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    os.chdir(frontend_dir)
    
    # Install Node.js dependencies
    if not run_command("npm install", "Installing Node.js dependencies", check=False):
        print("⚠️ npm install failed, trying with --legacy-peer-deps")
        if not run_command("npm install --legacy-peer-deps", "Installing Node.js dependencies (legacy)", check=False):
            print("❌ Failed to install frontend dependencies")
            os.chdir("..")
            return False
    
    os.chdir("..")
    print("✅ Frontend setup completed")
    return True

def create_launch_scripts():
    """Create convenient launch scripts"""
    print("📜 Creating launch scripts...")
    
    # Backend launch script
    if os.name == 'nt':  # Windows
        backend_script = """@echo off
echo 🚀 Starting Alpaca Domination Backend...
cd /d "%~dp0"
call alpaca_llm_env\\Scripts\\activate.bat
cd backend
python main.py
pause"""
        
        with open("start_backend.bat", 'w', encoding='utf-8') as f:
            f.write(backend_script)
        
        # Frontend launch script
        frontend_script = """@echo off
echo Starting Alpaca Domination Frontend...
cd /d "%~dp0\\frontend"
npm run dev
pause"""
        
        with open("start_frontend.bat", 'w', encoding='utf-8') as f:
            f.write(frontend_script)
        
        # Training script
        training_script = """@echo off
echo Starting Alpaca Domination Training...
cd /d "%~dp0"
call alpaca_llm_env\\Scripts\\activate.bat
python scripts\\train_model.py
pause"""
        
        with open("start_training.bat", 'w', encoding='utf-8') as f:
            f.write(training_script)
    
    else:  # Unix/Linux/macOS
        backend_script = """#!/bin/bash
echo "🚀 Starting Alpaca Domination Backend..."
cd "$(dirname "$0")"
source alpaca_llm_env/bin/activate
cd backend
python main.py"""
        
        with open("start_backend.sh", 'w') as f:
            f.write(backend_script)
        os.chmod("start_backend.sh", 0o755)
        
        # Frontend launch script
        frontend_script = """#!/bin/bash
echo "🎨 Starting Alpaca Domination Frontend..."
cd "$(dirname "$0")/frontend"
npm run dev"""
        
        with open("start_frontend.sh", 'w') as f:
            f.write(frontend_script)
        os.chmod("start_frontend.sh", 0o755)
        
        # Training script
        training_script = """#!/bin/bash
echo "🔥 Starting Alpaca Domination Training..."
cd "$(dirname "$0")"
source alpaca_llm_env/bin/activate
python scripts/train_model.py"""
        
        with open("start_training.sh", 'w') as f:
            f.write(training_script)
        os.chmod("start_training.sh", 0o755)
    
    print("✅ Launch scripts created")
    return True

def main():
    """Main setup function"""
    print("🏆 ALPACA DOMINATION PROJECT SETUP")
    print("=" * 50)
    print("🎯 Building the world's best Alpaca-trained language model")
    print("🚀 Targeting #1 global performance")
    print()
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Download dataset
    if download_dataset():
        success_count += 1
    
    # Step 2: Setup Python environment
    if setup_python_environment():
        success_count += 1
    
    # Step 3: Setup frontend (optional)
    print("🎨 Setting up frontend (optional)...")
    if Path("frontend/package.json").exists():
        if setup_frontend():
            success_count += 1
    else:
        print("⚠️ Frontend package.json not found, skipping frontend setup")
        success_count += 1
    
    # Step 4: Create launch scripts
    if create_launch_scripts():
        success_count += 1
    
    # Step 5: Final verification
    print("🔍 Verifying setup...")
    
    checks = [
        ("Data directory", Path("data").exists()),
        ("Training data", Path("data/alpaca_train.json").exists()),
        ("Python environment", Path("alpaca_llm_env").exists()),
        ("Model source", Path("src/models/alpaca_domination.py").exists()),
        ("Training script", Path("scripts/train_model.py").exists()),
        ("Backend", Path("backend/main.py").exists()),
    ]
    
    all_checks_passed = True
    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}: OK")
        else:
            print(f"❌ {check_name}: MISSING")
            all_checks_passed = False
    
    if all_checks_passed:
        success_count += 1
    
    # Summary
    print()
    print("🎉 SETUP SUMMARY")
    print("=" * 30)
    print(f"✅ Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("🏆 PERFECT SETUP! Ready to dominate!")
        print()
        print("🚀 Next steps:")
        print("1. Start training: python scripts/train_model.py")
        print("2. Start backend: python backend/main.py")
        print("3. Start frontend: cd frontend && npm run dev")
        print()
        print("🎯 Target: Achieve #1 Alpaca model globally!")
    else:
        print("⚠️  Some steps failed. Please check the errors above.")
        print("💡 You can retry individual steps or continue manually.")
    
    print()
    print("📚 Documentation: README.md")
    print("🆘 Support: Check the docs/ directory")
    print("🌟 Star the repo if this helps you build amazing AI!")

if __name__ == "__main__":
    main()
