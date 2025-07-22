from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import os
import json
import redis
from datetime import datetime
import asyncio
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Domain Finder API",
    description="AI-powered domain name suggestion API",
    version="1.0.0"
)

# CORS middleware for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your NextJS domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (now loaded from .env file)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
PORKBUN_API_KEY = os.getenv("PORKBUN_API_KEY", "")
PORKBUN_SECRET_KEY = os.getenv("PORKBUN_SECRET_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

print(f"Loaded Porkbun API Key: {'✅ Yes' if PORKBUN_API_KEY else '❌ No'}")
print(f"Loaded Porkbun Secret: {'✅ Yes' if PORKBUN_SECRET_KEY else '❌ No'}")

# Redis client with better error handling
def get_redis_client():
    try:
        client = redis.from_url(REDIS_URL)
        # Test the connection
        client.ping()
        print("✅ Redis connected successfully")
        return client
    except redis.ConnectionError:
        print("❌ Redis not available - running without caching")
        return None
    except Exception as e:
        print(f"❌ Redis error: {e} - running without caching")
        return None

redis_client = get_redis_client()

# Models
class DomainStyle(str, Enum):
    SHORT = "short"
    BRANDABLE = "brandable"
    KEYWORD = "keyword"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"

class DomainPreference(str, Enum):
    COM = ".com"
    NET = ".net"
    ORG = ".org"
    IO = ".io"
    ANY = "any"

class DomainRequest(BaseModel):
    idea: str = Field(..., description="Business idea or concept")
    field: str = Field(..., description="Industry or field")
    style: DomainStyle = Field(default=DomainStyle.BRANDABLE)
    domain_preference: DomainPreference = Field(default=DomainPreference.COM)
    max_price: float = Field(default=50.0, description="Maximum price per year")
    num_choices: int = Field(default=5, ge=1, le=20)

class DomainResult(BaseModel):
    domain: str
    available: bool
    price_first_year: Optional[float] = None
    price_annual: Optional[float] = None
    registrar: str = "porkbun"
    deal_info: Optional[str] = None
    score: float = Field(description="AI ranking score")
    pricing_details: Optional[Dict[str, Any]] = None

class DomainResponse(BaseModel):
    domains: List[DomainResult]
    request_id: str
    timestamp: datetime

# Domain suggestion AI agent using CrewAI concepts
class DomainSuggestionAgent:
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    async def generate_domain_ideas(self, request: DomainRequest) -> List[str]:
        """Generate domain name ideas using AI logic"""
        base_words = request.idea.lower().split()
        field_words = request.field.lower().split()
        
        suggestions = []
        
        # Style-based generation
        if request.style == DomainStyle.SHORT:
            # Generate short, catchy names
            suggestions.extend([
                f"{word[:4]}" for word in base_words + field_words
            ])
            suggestions.extend([
                f"{word1[:3]}{word2[:3]}" 
                for word1 in base_words for word2 in field_words
            ])
        
        elif request.style == DomainStyle.BRANDABLE:
            # Creative brandable names
            suffixes = ["ly", "fy", "io", "hub", "lab", "pro", "go"]
            prefixes = ["get", "my", "the", "smart", "quick", "easy"]
            
            for word in base_words + field_words:
                suggestions.extend([f"{word}{suffix}" for suffix in suffixes])
                suggestions.extend([f"{prefix}{word}" for prefix in prefixes])
        
        elif request.style == DomainStyle.KEYWORD:
            # Keyword-based combinations
            keywords = ["online", "digital", "web", "app", "tech", "service"]
            for word in base_words:
                suggestions.extend([f"{word}{keyword}" for keyword in keywords])
                suggestions.extend([f"{keyword}{word}" for keyword in keywords])
        
        elif request.style == DomainStyle.CREATIVE:
            # Creative variations
            variations = []
            for word in base_words + field_words:
                # Drop vowels
                no_vowels = ''.join([c for c in word if c not in 'aeiou'])
                if len(no_vowels) >= 3:
                    variations.append(no_vowels)
                
                # Add creative endings
                creative_endings = ["r", "d", "x", "z"]
                variations.extend([f"{word}{ending}" for ending in creative_endings])
            
            suggestions.extend(variations)
        
        elif request.style == DomainStyle.PROFESSIONAL:
            # Professional combinations
            prof_words = ["solutions", "services", "consulting", "group", "corp", "inc"]
            for word in base_words:
                suggestions.extend([f"{word}{prof}" for prof in prof_words])
        
        # Clean and filter suggestions
        suggestions = list(set([
            s for s in suggestions 
            if len(s) >= 3 and len(s) <= 15 and s.isalnum()
        ]))
        
        return suggestions[:3]  # Limit to 3 suggestions due to rate limiting

    async def check_domain_availability(self, domains: List[str], tld_preference: str = ".com") -> List[DomainResult]:
        """Check domain availability using Porkbun API with rate limiting"""
        results = []
        
        # Porkbun allows only 1 domain check per 10 seconds, so let's limit to 3 domains max
        # and add delays between requests
        limited_domains = domains[:3]  # Limit to 3 domains to avoid rate limiting
        
        for i, domain_name in enumerate(limited_domains):
            # Add TLD if not present
            if not any(domain_name.endswith(tld) for tld in ['.com', '.net', '.org', '.io', '.so', '.co']):
                # Convert enum to string and ensure it starts with a dot
                if hasattr(tld_preference, 'value'):
                    tld = tld_preference.value
                else:
                    tld = str(tld_preference)
                if not tld.startswith('.'):
                    tld = f".{tld}"
                domain_name = f"{domain_name}{tld}"
            
            try:
                # Check cache first
                cache_key = f"domain:{domain_name}"
                if redis_client:
                    try:
                        cached = redis_client.get(cache_key)
                        if cached:
                            result_data = json.loads(cached)
                            results.append(DomainResult(**result_data))
                            continue
                    except Exception as e:
                        print(f"Redis read error: {e}")
                
                # Query Porkbun API - FIXED: Using correct endpoint and payload
                async with httpx.AsyncClient() as client:
                    # Check domain availability using the correct endpoint
                    availability_response = await client.post(
                        f"https://api.porkbun.com/api/json/v3/domain/checkDomain/{domain_name}",
                        json={
                            "secretapikey": PORKBUN_SECRET_KEY,
                            "apikey": PORKBUN_API_KEY
                        },
                        headers={
                            "Content-Type": "application/json"
                        },
                        timeout=30.0
                    )
                    
                    # Debug the response
                    if availability_response.status_code != 200:
                        print(f"API Error for {domain_name}: Status {availability_response.status_code}")
                        print(f"Response: {availability_response.text}")
                        # Mark as unavailable if API error
                        is_available = False
                        availability_data = {}
                        price_first_year = None
                        price_annual = None
                        pricing_details = None
                        deal_info = None
                    else:
                        availability_data = availability_response.json()
                        print(f"API Success for {domain_name}: {availability_data}")
                        
                        # Parse the response correctly according to Porkbun docs
                        is_available = (
                            availability_data.get("status") == "SUCCESS" and 
                            availability_data.get("response", {}).get("avail") == "yes"
                        )
                        
                        price_first_year = None
                        price_annual = None
                        pricing_details = None
                        deal_info = None
                        
                        if is_available and availability_data.get("response"):
                            response_data = availability_data["response"]
                            
                            # Get pricing from the domain check response
                            price_first_year = float(response_data.get("price", 0))
                            price_annual = float(response_data.get("regularPrice", 0))
                            
                            # Check for additional pricing details
                            additional = response_data.get("additional", {})
                            if additional:
                                renewal_price = additional.get("renewal", {}).get("price")
                                transfer_price = additional.get("transfer", {}).get("price")
                                
                                pricing_details = {
                                    "registration": price_first_year,
                                    "renewal": float(renewal_price) if renewal_price else price_annual,
                                    "transfer": float(transfer_price) if transfer_price else price_annual,
                                    "premium": response_data.get("premium") == "yes",
                                    "first_year_promo": response_data.get("firstYearPromo") == "yes"
                                }
                                
                                # Set annual price to renewal price if different
                                if renewal_price:
                                    price_annual = float(renewal_price)
                            
                            # Format deal information
                            if price_first_year != price_annual and price_annual > 0:
                                deal_info = f"First year: ${price_first_year:.2f}, Then: ${price_annual:.2f}/year"
                            elif response_data.get("firstYearPromo") == "yes":
                                deal_info = f"Promotional pricing: ${price_first_year:.2f} first year"
                    
                    result = DomainResult(
                        domain=domain_name,
                        available=is_available,
                        price_first_year=price_first_year,
                        price_annual=price_annual,
                        registrar="porkbun",
                        deal_info=deal_info,
                        pricing_details=pricing_details,
                        score=self.calculate_domain_score(domain_name, is_available, price_first_year)
                    )
                    
                    # Cache result with smart expiration
                    if redis_client:
                        try:
                            # Cache available domains for 2 hours, taken domains for 24 hours
                            expiry = 7200 if is_available else 86400
                            redis_client.setex(
                                cache_key, expiry, 
                                json.dumps(result.dict())
                            )
                        except Exception as e:
                            print(f"Redis write error: {e}")
                    
                    results.append(result)
                    
                    # Add delay between requests to respect rate limits (except for last request)
                    if i < len(limited_domains) - 1:
                        await asyncio.sleep(10)  # Wait 10 seconds between requests
                
            except Exception as e:
                logging.error(f"Error checking domain {domain_name}: {e}")
                # Add as unavailable if error
                results.append(DomainResult(
                    domain=domain_name,
                    available=False,
                    score=0.0,
                    registrar="porkbun"
                ))
        
        return results

    def calculate_domain_score(self, domain: str, is_available: bool, price: Optional[float] = None) -> float:
        """Calculate AI-based domain score"""
        score = 0.0
        
        if not is_available:
            return 0.0
        
        # Length score (shorter is better)
        length = len(domain.split('.')[0])
        if length <= 6:
            score += 3.0
        elif length <= 10:
            score += 2.0
        elif length <= 15:
            score += 1.0
        
        # TLD score
        if domain.endswith('.com'):
            score += 2.0
        elif domain.endswith('.io'):
            score += 1.5
        elif domain.endswith('.net'):
            score += 1.0
        
        # Price score (lower price is better)
        if price is not None:
            if price <= 15:
                score += 1.5
            elif price <= 25:
                score += 1.0
            elif price <= 50:
                score += 0.5
        
        # Readability (no numbers, hyphens)
        domain_name = domain.split('.')[0]
        if domain_name.isalpha():
            score += 1.0
        
        # Brandability (vowel-consonant patterns)
        vowels = sum(1 for c in domain_name if c in 'aeiou')
        consonants = len(domain_name) - vowels
        if vowels > 0 and consonants > 0:
            score += 1.0
        
        return min(score, 5.0)

# Initialize AI agent
domain_agent = DomainSuggestionAgent()

# Test Porkbun API connection
@app.get("/api/test-porkbun")
async def test_porkbun_connection():
    """Test Porkbun API connection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.porkbun.com/api/json/v3/ping",
                json={
                    "secretapikey": PORKBUN_SECRET_KEY,
                    "apikey": PORKBUN_API_KEY
                },
                headers={
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "porkbun_response": data,
                    "message": "Porkbun API connection successful"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Porkbun API returned status {response.status_code}",
                    "response": response.text
                }
                
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to Porkbun API: {str(e)}"
        }

# API Routes
@app.post("/api/domains/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """Main endpoint: suggest domain names"""
    
    try:
        # Generate domain suggestions using AI
        domain_ideas = await domain_agent.generate_domain_ideas(request)
        
        # Check availability
        domain_results = await domain_agent.check_domain_availability(domain_ideas, request.domain_preference)
        
        # Filter available domains and sort by score
        available_domains = [d for d in domain_results if d.available]
        available_domains.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to requested number
        final_domains = available_domains[:request.num_choices]
        
        # Generate response
        response = DomainResponse(
            domains=final_domains,
            request_id=f"req_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating domain suggestions: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/api/pricing")
async def get_pricing():
    """Get pricing information"""
    return {
        "status": "open_beta",
        "message": "API is currently free during development phase",
        "future_pricing": {
            "free_requests": 2,
            "price_per_5_requests": 0.20,
            "currency": "USD"
        }
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )