import os
from groq import Groq
from dotenv import load_dotenv

# 1. Load the secret API key from your local .env file configuration
load_dotenv()

# 2. Extract the API key securely from memory
api_key_value = os.getenv("GROQ_API_KEY")

# 3. Explicitly initialize the client using the extracted key
client = Groq(api_key=api_key_value)

try:
    print("⏳ Sending test chunk to Groq...")
    
    completion = client.chat.completions.create(
        # FIX: Swapped out retired model string for active production engine
        model="openai/gpt-oss-20b", 
        
        messages=[
            {
                "role": "user", 
                "content": "Extract keywords from: Revenue grew by 15% to ₹12,000 Crore."
            }
        ],
        temperature=0.1
    )
    
    print("\n🚀 GROQ RESPONSE SUCCESSFUL:")
    print("=" * 40)
    print(completion.choices[0].message.content.strip())
    print("=" * 40)

except Exception as e:
    print(f"\n❌ Configuration Error: {str(e)}")
