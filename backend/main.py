"""
🤖 Your Custom Trained AI Chat Model
FastAPI server for your Alpaca-trained language model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import uuid
from datetime import datetime
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model():
    """Load YOUR trained AI model"""
    global model, tokenizer
    
    # Try different possible paths for the model
    possible_paths = [
        Path("ultra_fast_model"),
        Path("../ultra_fast_model"),
        Path("../../ultra_fast_model")
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        logger.error("❌ Your trained model not found in any expected location!")
        return False
    
    try:
        logger.info("🔄 Loading YOUR trained AI model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"✅ YOUR AI MODEL loaded successfully on {device}!")
        logger.info(f"📊 Model parameters: {model.num_parameters():,}")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading your model: {e}")
        return False

# Load YOUR model on startup
model_loaded = load_trained_model()

# FastAPI app
app = FastAPI(
    title="Your Custom AI Chat Model",
    description="Your personally trained Alpaca AI model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 150

def generate_ai_response(messages: List[ChatMessage], max_tokens: int = 150) -> str:
    """Generate response using YOUR trained AI model"""
    global model, tokenizer
    
    if not model or not tokenizer:
        return "❌ Your AI model is not loaded. Please check the model files."
    
    try:
        # Get user message
        user_content = messages[-1].content if messages else "Hello"
        
        # Format in Alpaca instruction style (how your model was trained)
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_content}\n\n### Response:\n"
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate response using YOUR trained model
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the AI response part
        prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = full_response[len(prompt_decoded):].strip()
        
        return response if response else "I understand your request. How can I help you further?"
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return f"Error in AI generation: {str(e)}"

# API Routes
@app.get("/")
async def root():
    """Status of your AI model"""
    return {
        "message": "🤖 Your Custom Trained AI Model",
        "model_type": "Alpaca-trained GPT-2",
        "model_loaded": model_loaded,
        "device": device,
        "parameters": "124M (trained on 52K samples)",
        "endpoints": {
            "chat_interface": "/chat",
            "api": "/v1/chat/completions",
            "docs": "/docs"
        }
    }

@app.post("/v1/chat/completions")
async def chat_with_your_ai(request: ChatRequest):
    """Chat with YOUR trained AI model"""
    try:
        response_text = generate_ai_response(request.messages, request.max_tokens or 150)
        
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": "your-custom-alpaca-ai",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.messages[-1].content.split()) if request.messages else 0,
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.messages[-1].content.split()) + len(response_text.split()) if request.messages else len(response_text.split())
            }
        }
    
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat", response_class=HTMLResponse)
async def your_ai_chat_interface():
    """Chat with YOUR trained AI model"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Your Custom AI Chat</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .header { 
                text-align: center; 
                margin-bottom: 20px; 
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .status { 
                padding: 15px; 
                background: linear-gradient(45deg, #56ab2f, #a8e6cf);
                color: white;
                margin-bottom: 15px; 
                border-radius: 10px; 
                text-align: center;
                font-weight: bold;
            }
            .chat-container { 
                border: 2px solid #f0f0f0;
                height: 400px; 
                overflow-y: auto; 
                padding: 15px; 
                margin-bottom: 15px; 
                background: #fafafa;
                border-radius: 10px;
            }
            .message { 
                margin: 15px 0; 
                padding: 12px 15px; 
                border-radius: 18px; 
                max-width: 80%; 
                word-wrap: break-word;
            }
            .user { 
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; 
                margin-left: auto; 
                text-align: right;
            }
            .assistant { 
                background: linear-gradient(45deg, #f093fb, #f5576c);
                color: white; 
                margin-right: auto; 
            }
            .input-container { 
                display: flex; 
                gap: 10px; 
                background: white;
                padding: 10px;
                border-radius: 25px;
                border: 2px solid #e0e0e0;
            }
            input[type="text"] { 
                flex: 1; 
                padding: 12px 15px; 
                border: none;
                border-radius: 20px; 
                font-size: 16px;
                outline: none;
                background: #f8f9fa;
            }
            button { 
                padding: 12px 25px; 
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white; 
                border: none; 
                border-radius: 20px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover { 
                transform: scale(1.05);
            }
            button:disabled {
                background: #ccc;
                transform: none;
                cursor: not-allowed;
            }
            .typing {
                opacity: 0.7;
                font-style: italic;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Your Custom Trained AI</h1>
                <p>Alpaca Model - 124M Parameters - Trained on 52K Samples</p>
            </div>
            
            <div class="status" id="status">
                <span id="model-status">🔄 Checking your AI model...</span>
            </div>
            
            <div class="chat-container" id="chat"></div>
            
            <div class="input-container">
                <input type="text" id="message-input" placeholder="Ask your AI anything..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" id="send-btn">Send</button>
            </div>
        </div>

        <script>
            let chatContainer = document.getElementById('chat');
            let messageInput = document.getElementById('message-input');
            let statusElement = document.getElementById('model-status');
            let sendButton = document.getElementById('send-btn');

            // Check YOUR AI model status
            fetch('/')
                .then(response => response.json())
                .then(data => {
                    statusElement.innerHTML = data.model_loaded ? 
                        '✅ Your AI Model is Ready! 🤖' : 
                        '❌ Your AI Model is not loaded';
                    if (!data.model_loaded) {
                        addMessage('Your AI model is not loaded. Please check the training.', 'assistant');
                    }
                });

            function addMessage(content, role) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                messageDiv.innerHTML = role === 'user' ? 
                    `<strong>You:</strong> ${content}` : 
                    `<strong>🤖 Your AI:</strong> ${content}`;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function setLoading(loading) {
                sendButton.disabled = loading;
                sendButton.innerHTML = loading ? '🤔 Thinking...' : 'Send';
                messageInput.disabled = loading;
            }

            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                addMessage(message, 'user');
                messageInput.value = '';
                setLoading(true);

                try {
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: [{ role: 'user', content: message }],
                            max_tokens: 150
                        })
                    });

                    const data = await response.json();
                    if (data.choices && data.choices[0]) {
                        const aiResponse = data.choices[0].message.content;
                        addMessage(aiResponse, 'assistant');
                    } else {
                        addMessage('Sorry, I could not generate a response.', 'assistant');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('Error: Could not connect to your AI model.', 'assistant');
                } finally {
                    setLoading(false);
                }
            }

            // Welcome message from YOUR AI
            addMessage('Hello! I\\'m your custom trained AI model. I was trained on 52,000 Alpaca instruction-response pairs. Ask me anything!', 'assistant');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("🤖 Starting YOUR Custom AI Chat Model...")
    print("🎯 Model: Alpaca-trained GPT-2 (124M parameters)")
    print("📊 Training: 52K instruction-response pairs")
    print("🌐 Chat Interface: http://localhost:8000/chat")
    print("🔌 API: http://localhost:8000/v1/chat/completions")
    print("📚 Documentation: http://localhost:8000/docs")
    print("⚡ Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
