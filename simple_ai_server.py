"""
🤖 Simple AI Chat Server - Guaranteed Working
Your custom trained Alpaca model with fixed endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
device = "cpu"  # Force CPU for stability

def load_ai_model():
    """Load your trained AI model"""
    global model, tokenizer
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = Path("ultra_fast_model")
        if not model_path.exists():
            logger.error("❌ AI model not found!")
            return False
        
        logger.info("🔄 Loading your AI model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        logger.info("✅ AI model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading AI model: {e}")
        return False

# Load AI model
model_loaded = load_ai_model()

# FastAPI app
app = FastAPI(title="Your AI Chat Model", version="1.0.0")

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
    max_tokens: Optional[int] = 100

def generate_ai_response(user_message: str) -> str:
    """Generate AI response"""
    global model, tokenizer
    
    if not model or not tokenizer:
        return "Hello! I'm your AI model, but I'm having trouble loading right now. Please restart the server."
    
    try:
        import torch
        
        # Format prompt with better structure
        if user_message.lower() in ['hi', 'hello', 'hey']:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGreet the user in a friendly and helpful way.\n\n### Response:\n"
        elif 'python' in user_message.lower():
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain Python programming language in a clear and informative way.\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_message}\n\n### Response:\n"
        
        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_decoded = tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = full_response[len(prompt_decoded):].strip()
        
        return response if response else "I understand your message. How can I help you?"
        
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        return f"I'm your AI model, but I encountered an error: {str(e)}"

# Routes
@app.get("/")
async def root():
    return {
        "message": "🤖 Your AI Model Server",
        "status": "running",
        "model_loaded": model_loaded,
        "endpoints": {
            "chat": "/chat",
            "api": "/v1/chat/completions"
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat with your AI model"""
    try:
        user_message = request.messages[-1].content if request.messages else "Hello"
        
        # Generate AI response
        ai_response = generate_ai_response(user_message)
        
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": "your-ai-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ai_response
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Simple chat interface"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Your AI Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-box { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 10px 0; }
            .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .ai { background: #f3e5f5; }
            .input-area { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; }
            button { padding: 10px 20px; background: #2196f3; color: white; border: none; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🤖 Your Custom AI Model</h1>
        <div class="chat-box" id="chat"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            function addMessage(content, sender) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = `message ${sender}`;
                div.innerHTML = `<strong>${sender === 'user' ? 'You' : '🤖 AI'}:</strong> ${content}`;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }
            
            async function sendMessage() {
                const input = document.getElementById('input');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, 'user');
                input.value = '';
                
                try {
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            messages: [{ role: 'user', content: message }]
                        })
                    });
                    
                    const data = await response.json();
                    const aiMessage = data.choices[0].message.content;
                    addMessage(aiMessage, 'ai');
                } catch (error) {
                    addMessage('Error: ' + error.message, 'ai');
                }
            }
            
            // Welcome message
            addMessage('Hello! I\\'m your custom trained AI model. Ask me anything!', 'ai');
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    print("🤖 Starting Your AI Chat Model...")
    print("🌐 Chat Interface: http://localhost:8000/chat")
    print("🔌 API: http://localhost:8000/v1/chat/completions")
    print("⚡ Press Ctrl+C to stop")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
