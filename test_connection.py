import requests
import json

def test_ai_connection():
    try:
        print("🔄 Testing AI model connection...")
        
        # Test 1: Check if server is running
        response = requests.get("http://localhost:8000/")
        print(f"✅ Server Status: {response.status_code}")
        print(f"📊 Server Response: {response.json()}")
        
        # Test 2: Test chat API
        chat_data = {
            "messages": [{"role": "user", "content": "Hello, are you working?"}],
            "max_tokens": 50
        }
        
        print("\n🤖 Testing chat API...")
        chat_response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=chat_data
        )
        
        print(f"✅ Chat API Status: {chat_response.status_code}")
        
        if chat_response.status_code == 200:
            result = chat_response.json()
            if "choices" in result and result["choices"]:
                ai_message = result["choices"][0]["message"]["content"]
                print(f"🤖 AI Response: {ai_message}")
            else:
                print("❌ No AI response in result")
                print(f"Raw result: {result}")
        else:
            print(f"❌ Chat API Error: {chat_response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to AI server. Is it running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ai_connection()
