#!/usr/bin/env python3
"""
🧪 Terminal Model Tester
Test your trained model directly from the terminal
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

def load_model():
    """Load the trained model"""
    model_path = Path("ultra_fast_model")
    
    if not model_path.exists():
        print("❌ Model not found!")
        print("Please run training first using:")
        print("  .\RUN_PROJECT.bat -> Option 1 (Ultra-Fast Training)")
        return None, None
    
    print("🔄 Loading trained model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print(f"✅ Model loaded successfully!")
        print(f"📊 Parameters: {model.num_parameters():,}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_tokens=100):
    """Generate response for a prompt"""
    # Format in Alpaca style
    formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_tokens,
            num_beams=2,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
    response = full_response[len(prompt_decoded):].strip()
    
    return response if response else "I understand your request. Could you please provide more details?"

def run_test_prompts(model, tokenizer):
    """Run predefined test prompts"""
    test_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "Write a short story about a robot.",
        "Explain the concept of democracy.",
        "What is the difference between Python and JavaScript?",
        "How does photosynthesis work?",
        "What are the main causes of climate change?",
        "Describe the process of protein synthesis."
    ]
    
    print("\n🧪 Running Test Prompts:")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}/10: {prompt}")
        print("🤖 Response:", end=" ")
        
        response = generate_response(model, tokenizer, prompt)
        print(response)
        print("-" * 40)

def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n💬 Interactive Mode - Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n🧑 You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("🤖 Alpaca: ", end="")
            response = generate_response(model, tokenizer, prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    print("🚀 ALPACA DOMINATION TERMINAL TESTER")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model()
    if not model:
        return
    
    # Menu
    while True:
        print("\nChoose an option:")
        print("1. 🧪 Run Test Prompts (10 predefined questions)")
        print("2. 💬 Interactive Chat Mode")
        print("3. 🔧 Single Prompt Test")
        print("4. ❌ Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_test_prompts(model, tokenizer)
        elif choice == "2":
            interactive_mode(model, tokenizer)
        elif choice == "3":
            prompt = input("\nEnter your prompt: ").strip()
            if prompt:
                print("🤖 Response:", generate_response(model, tokenizer, prompt))
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
