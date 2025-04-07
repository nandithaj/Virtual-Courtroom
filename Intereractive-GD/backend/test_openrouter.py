from openai import OpenAI
import os
from dotenv import load_dotenv

def test_openrouter_connection():
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            print("Error: OPENROUTER_API_KEY not found in environment variables")
            return False
            
        # Initialize OpenAI client with OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Test API call
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "Interactive GD",
            },
            model="google/gemini-pro",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
            ]
        )
        
        print("API Connection Successful!")
        print("Response:", completion.choices[0].message.content)
        return True
        
    except Exception as e:
        print(f"Error testing OpenRouter API: {str(e)}")
        return False

if __name__ == "__main__":
    test_openrouter_connection() 