# Save as test_porkbun.py and run it
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

PORKBUN_API_KEY = os.getenv("PORKBUN_API_KEY", "")
PORKBUN_SECRET_KEY = os.getenv("PORKBUN_SECRET_KEY", "")

async def test_porkbun_api():
    print(f"API Key: {PORKBUN_API_KEY[:10]}..." if PORKBUN_API_KEY else "No API Key")
    print(f"Secret Key: {PORKBUN_SECRET_KEY[:10]}..." if PORKBUN_SECRET_KEY else "No Secret Key")
    
    # Test simple domain availability 
    async with httpx.AsyncClient() as client:
        try:
            print("\n🧪 Testing domain availability...")
            test_domain = "thisisadefinitelyavailabledomain12345.com"
            
            response = await client.post(
                "https://porkbun.com/api/json/v3/domain/isAvailable",
                json={
                    "apikey": PORKBUN_API_KEY,
                    "secretapikey": PORKBUN_SECRET_KEY,
                    "domain": test_domain
                }
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Raw Response: '{response.text}'")
            
            if response.text:
                try:
                    data = response.json()
                    print(f"JSON Response: {data}")
                except:
                    print("Response is not valid JSON")
            else:
                print("Response is completely empty!")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_porkbun_api())