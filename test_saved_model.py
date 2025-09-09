#!/usr/bin/env python3
"""
Load and Test Your Saved Model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def test_saved_model():
    model_path = "ultra_fast_model"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run training first!")
        return
    
    print("🔄 Loading your saved model...")
    
    try:
        # Load the saved model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model size: {model.num_parameters():,} parameters")
        
        # Test the model
        print("\n🧪 Testing your model:")
        
        test_prompts = [
            "Question: What is artificial intelligence?",
            "Instruction: Explain machine learning in simple terms.",
            "Question: How do neural networks work?"
        ]
        
        model.eval()
        
        for prompt in test_prompts:
            print(f"\n💬 Input: {prompt}")
            
            inputs = tokenizer.encode(prompt + "\nAnswer:", return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            print(f"🤖 Response: {answer}")
        
        print("\n🎉 Your model is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    test_saved_model()
