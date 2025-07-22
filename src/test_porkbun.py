import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

PORKBUN_API_KEY = os.getenv("PORKBUN_API_KEY", "")
PORKBUN_SECRET_KEY = os.getenv("PORKBUN_SECRET_KEY", "")

async def test_porkbun_detailed():
    print(f"Testing with API Key: {PORKBUN_API_KEY}")
    print(f"Testing with Secret Key: {PORKBUN_SECRET_KEY}")
    print(f"Key lengths: API={len(PORKBUN_API_KEY)}, Secret={len(PORKBUN_SECRET_KEY)}")
    
    # Test 1: Try the ping endpoint (simpler)
    async with httpx.AsyncClient() as client:
        try:
            print("\n🧪 Test 1: Basic API Ping...")
            response = await client.post(
                "https://api.porkbun.com/api/json/v3/ping",
                json={
                    "apikey": PORKBUN_API_KEY,
                    "secretapikey": PORKBUN_SECRET_KEY
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
        except Exception as e:
            print(f"❌ Ping failed: {e}")
    
    # Test 2: Try pricing endpoint (doesn't require domain ownership)
    async with httpx.AsyncClient() as client:
        try:
            print("\n🧪 Test 2: Pricing Endpoint...")
            response = await client.post(
                "https://api.porkbun.com/api/json/v3/pricing/get",
                json={
                    "apikey": PORKBUN_API_KEY,
                    "secretapikey": PORKBUN_SECRET_KEY
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
        except Exception as e:
            print(f"❌ Pricing failed: {e}")
    
    # Test 3: Try the correct domain check endpoint
    async with httpx.AsyncClient() as client:
        try:
            print("\n🧪 Test 3: Correct Domain Check Endpoint...")
            test_domain = "haloway.co"
            response = await client.post(
                f"https://api.porkbun.com/api/json/v3/domain/checkDomain/{test_domain}",
                json={
                    "apikey": PORKBUN_API_KEY,
                    "secretapikey": PORKBUN_SECRET_KEY
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
        except Exception as e:
            print(f"❌ Domain check failed: {e}")

    # Test 4: Check if it's a rate limiting issue
    print("\n🧪 Test 4: Check Response Headers...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.porkbun.com/api/json/v3/ping",
                json={
                    "apikey": PORKBUN_API_KEY,
                    "secretapikey": PORKBUN_SECRET_KEY
                }
            )
            
            print("Full Headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"❌ Header check failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_porkbun_detailed())