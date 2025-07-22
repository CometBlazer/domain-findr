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
import base64

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Domain Finder API",
    description="AI-powered domain name suggestion API with multi-provider support",
    version="2.0.0"
)

# CORS middleware for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your NextJS domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
PORKBUN_API_KEY = os.getenv("PORKBUN_API_KEY", "")
PORKBUN_SECRET_KEY = os.getenv("PORKBUN_SECRET_KEY", "")
NAMECOM_API_TOKEN = os.getenv("NAMECOM_API_TOKEN", "")
NAMECOM_USERNAME = os.getenv("NAMECOM_USERNAME", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

print(f"Loaded Porkbun API Key: {'✅ Yes' if PORKBUN_API_KEY else '❌ No'}")
print(f"Loaded Porkbun Secret: {'✅ Yes' if PORKBUN_SECRET_KEY else '❌ No'}")
print(f"Loaded Name.com Token: {'✅ Yes' if NAMECOM_API_TOKEN else '❌ No'}")
print(f"Loaded Name.com Username: {'✅ Yes' if NAMECOM_USERNAME else '❌ No'}")

# Redis client with better error handling
def get_redis_client():
    try:
        client = redis.from_url(REDIS_URL)
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
    CO = ".co"
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
    registrar: str
    deal_info: Optional[str] = None
    score: float = Field(description="AI ranking score")
    pricing_details: Optional[Dict[str, Any]] = None

class DomainResponse(BaseModel):
    domains: List[DomainResult]
    request_id: str
    timestamp: datetime
    search_summary: Dict[str, Any]

# Base Domain Provider class
class BaseDomainProvider:
    def __init__(self, name: str):
        self.name = name
    
    async def check_domains(self, domains: List[str], tld_preference: str) -> List[DomainResult]:
        raise NotImplementedError
    
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

# Porkbun Provider
class PorkbunProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("porkbun")
    
    async def check_domains(self, domains: List[str], tld_preference: str) -> List[DomainResult]:
        """Check domain availability using Porkbun API with rate limiting"""
        results = []
        
        # Porkbun: 1 domain per 10 seconds - very limited
        limited_domains = domains[:2]  # Limit to 2 domains for Porkbun
        
        for i, domain_name in enumerate(limited_domains):
            # Add TLD if not present
            if not any(domain_name.endswith(tld) for tld in ['.com', '.net', '.org', '.io', '.co']):
                if hasattr(tld_preference, 'value'):
                    tld = tld_preference.value
                else:
                    tld = str(tld_preference)
                if not tld.startswith('.'):
                    tld = f".{tld}"
                domain_name = f"{domain_name}{tld}"
            
            try:
                # Check cache first
                cache_key = f"porkbun:domain:{domain_name}"
                if redis_client:
                    try:
                        cached = redis_client.get(cache_key)
                        if cached:
                            result_data = json.loads(cached)
                            results.append(DomainResult(**result_data))
                            continue
                    except Exception as e:
                        print(f"Redis read error: {e}")
                
                async with httpx.AsyncClient() as client:
                    availability_response = await client.post(
                        f"https://api.porkbun.com/api/json/v3/domain/checkDomain/{domain_name}",
                        json={
                            "secretapikey": PORKBUN_SECRET_KEY,
                            "apikey": PORKBUN_API_KEY
                        },
                        headers={"Content-Type": "application/json"},
                        timeout=30.0
                    )
                    
                    if availability_response.status_code != 200:
                        print(f"Porkbun API Error for {domain_name}: {availability_response.status_code}")
                        print(f"Response: {availability_response.text}")
                        continue
                    
                    availability_data = availability_response.json()
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
                        price_first_year = float(response_data.get("price", 0))
                        price_annual = float(response_data.get("regularPrice", 0))
                        
                        additional = response_data.get("additional", {})
                        if additional:
                            renewal_price = additional.get("renewal", {}).get("price")
                            if renewal_price:
                                price_annual = float(renewal_price)
                            
                            pricing_details = {
                                "registration": price_first_year,
                                "renewal": price_annual,
                                "premium": response_data.get("premium") == "yes",
                                "first_year_promo": response_data.get("firstYearPromo") == "yes"
                            }
                        
                        if price_first_year != price_annual and price_annual > 0:
                            deal_info = f"First year: ${price_first_year:.2f}, Then: ${price_annual:.2f}/year"
                    
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
                    
                    # Cache result
                    if redis_client:
                        try:
                            expiry = 7200 if is_available else 86400
                            redis_client.setex(cache_key, expiry, json.dumps(result.dict()))
                        except Exception as e:
                            print(f"Redis write error: {e}")
                    
                    results.append(result)
                    
                    # Rate limiting: wait between requests
                    if i < len(limited_domains) - 1:
                        await asyncio.sleep(10)
                
            except Exception as e:
                logging.error(f"Porkbun error checking domain {domain_name}: {e}")
                continue
        
        return results

# Name.com Provider  
class NameComProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("name.com")
    
    async def check_domains(self, domains: List[str], tld_preference: str) -> List[DomainResult]:
        """Check domain availability using Name.com API"""
        results = []
        
        # Name.com allows up to 50 domains per call - much better!
        limited_domains = domains[:10]  # Check up to 10 domains
        
        # Prepare domain list with TLD
        domain_list = []
        for domain_name in limited_domains:
            if not any(domain_name.endswith(tld) for tld in ['.com', '.net', '.org', '.io', '.co']):
                if hasattr(tld_preference, 'value'):
                    tld = tld_preference.value
                else:
                    tld = str(tld_preference)
                if not tld.startswith('.'):
                    tld = f".{tld}"
                domain_name = f"{domain_name}{tld}"
            domain_list.append(domain_name)
        
        try:
            # Create Basic Auth header
            auth_string = f"{NAMECOM_USERNAME}:{NAMECOM_API_TOKEN}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            async with httpx.AsyncClient() as client:
                # Name.com bulk availability check
                availability_response = await client.post(
                    "https://api.name.com/v4/domains:checkAvailability",
                    json={"domainNames": domain_list},
                    headers={
                        "Authorization": f"Basic {auth_b64}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if availability_response.status_code != 200:
                    print(f"Name.com API Error: {availability_response.status_code}")
                    print(f"Response: {availability_response.text}")
                    return results
                
                availability_data = availability_response.json()
                
                # Process results
                for domain_info in availability_data.get("results", []):
                    domain_name = domain_info.get("domainName", "")
                    is_available = domain_info.get("purchasable", False)
                    
                    price_first_year = None
                    price_annual = None
                    pricing_details = None
                    deal_info = None
                    
                    if is_available:
                        price_first_year = domain_info.get("purchasePrice", 0) / 100  # Convert from cents
                        price_annual = domain_info.get("renewalPrice", 0) / 100      # Convert from cents
                        
                        pricing_details = {
                            "registration": price_first_year,
                            "renewal": price_annual,
                            "premium": domain_info.get("premium", False)
                        }
                        
                        if price_first_year != price_annual and price_annual > 0:
                            deal_info = f"First year: ${price_first_year:.2f}, Then: ${price_annual:.2f}/year"
                    
                    result = DomainResult(
                        domain=domain_name,
                        available=is_available,
                        price_first_year=price_first_year,
                        price_annual=price_annual,
                        registrar="name.com",
                        deal_info=deal_info,
                        pricing_details=pricing_details,
                        score=self.calculate_domain_score(domain_name, is_available, price_first_year)
                    )
                    
                    # Cache result
                    if redis_client:
                        try:
                            cache_key = f"namecom:domain:{domain_name}"
                            expiry = 7200 if is_available else 86400
                            redis_client.setex(cache_key, expiry, json.dumps(result.dict()))
                        except Exception as e:
                            print(f"Redis write error: {e}")
                    
                    results.append(result)
                
        except Exception as e:
            logging.error(f"Name.com error checking domains: {e}")
        
        return results

# Domain suggestion AI agent
class DomainSuggestionAgent:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.porkbun = PorkbunProvider()
        self.namecom = NameComProvider()
    
    async def generate_domain_ideas(self, request: DomainRequest) -> List[str]:
        """Generate domain name ideas using AI logic"""
        base_words = request.idea.lower().split()
        field_words = request.field.lower().split()
        
        suggestions = []
        
        # Style-based generation
        if request.style == DomainStyle.SHORT:
            suggestions.extend([f"{word[:4]}" for word in base_words + field_words])
            suggestions.extend([f"{word1[:3]}{word2[:3]}" for word1 in base_words for word2 in field_words])
        
        elif request.style == DomainStyle.BRANDABLE:
            suffixes = ["ly", "fy", "hub", "lab", "pro", "go", "kit", "box"]
            prefixes = ["get", "my", "the", "smart", "quick", "easy", "auto", "super"]
            
            for word in base_words + field_words:
                suggestions.extend([f"{word}{suffix}" for suffix in suffixes])
                suggestions.extend([f"{prefix}{word}" for prefix in prefixes])
        
        elif request.style == DomainStyle.KEYWORD:
            keywords = ["online", "digital", "web", "app", "tech", "service", "cloud", "ai"]
            for word in base_words:
                suggestions.extend([f"{word}{keyword}" for keyword in keywords])
                suggestions.extend([f"{keyword}{word}" for keyword in keywords])
        
        elif request.style == DomainStyle.CREATIVE:
            variations = []
            for word in base_words + field_words:
                no_vowels = ''.join([c for c in word if c not in 'aeiou'])
                if len(no_vowels) >= 3:
                    variations.append(no_vowels)
                
                creative_endings = ["r", "d", "x", "z", "y"]
                variations.extend([f"{word}{ending}" for ending in creative_endings])
            
            suggestions.extend(variations)
        
        elif request.style == DomainStyle.PROFESSIONAL:
            prof_words = ["solutions", "services", "consulting", "group", "corp", "systems"]
            for word in base_words:
                suggestions.extend([f"{word}{prof}" for prof in prof_words])
        
        # Clean and filter suggestions
        suggestions = list(set([
            s for s in suggestions 
            if len(s) >= 3 and len(s) <= 15 and s.isalnum()
        ]))
        
        # Generate more suggestions since we have two providers
        return suggestions[:15]  # Generate up to 15 suggestions

    async def search_domains_parallel(self, domains: List[str], request: DomainRequest) -> DomainResponse:
        """Search domains across multiple providers in parallel"""
        
        # Run both providers in parallel
        tasks = []
        
        if PORKBUN_API_KEY and PORKBUN_SECRET_KEY:
            tasks.append(self.porkbun.check_domains(domains, request.domain_preference))
        
        if NAMECOM_API_TOKEN and NAMECOM_USERNAME:
            tasks.append(self.namecom.check_domains(domains, request.domain_preference))
        
        if not tasks:
            raise HTTPException(status_code=500, detail="No domain providers configured")
        
        # Execute searches in parallel
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results from all providers
        all_results = []
        search_summary = {
            "providers_used": [],
            "total_domains_checked": 0,
            "available_domains_found": 0,
            "errors": []
        }
        
        for i, results in enumerate(provider_results):
            if isinstance(results, Exception):
                provider_name = ["porkbun", "name.com"][i] if len(tasks) > 1 else "porkbun"
                search_summary["errors"].append(f"{provider_name}: {str(results)}")
                continue
            
            if results:
                provider_name = results[0].registrar if results else "unknown"
                search_summary["providers_used"].append(provider_name)
                search_summary["total_domains_checked"] += len(results)
                
                available_results = [r for r in results if r.available]
                search_summary["available_domains_found"] += len(available_results)
                all_results.extend(available_results)
        
        # Remove duplicates (same domain from different providers)
        seen_domains = set()
        unique_results = []
        for result in all_results:
            if result.domain not in seen_domains:
                seen_domains.add(result.domain)
                unique_results.append(result)
        
        # Sort by score (best domains first)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to requested number
        final_domains = unique_results[:request.num_choices]
        
        return DomainResponse(
            domains=final_domains,
            request_id=f"req_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            search_summary=search_summary
        )

# Initialize AI agent
domain_agent = DomainSuggestionAgent()

# Test endpoints
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
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"status": "success", "provider": "porkbun", "response": response.json()}
            else:
                return {"status": "error", "provider": "porkbun", "message": response.text}
                
    except Exception as e:
        return {"status": "error", "provider": "porkbun", "message": str(e)}

@app.get("/api/test-namecom")
async def test_namecom_connection():
    """Test Name.com API connection"""
    try:
        auth_string = f"{NAMECOM_USERNAME}:{NAMECOM_API_TOKEN}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.name.com/v4/hello",
                headers={"Authorization": f"Basic {auth_b64}"},
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"status": "success", "provider": "name.com", "response": response.json()}
            else:
                return {"status": "error", "provider": "name.com", "message": response.text}
                
    except Exception as e:
        return {"status": "error", "provider": "name.com", "message": str(e)}

# Main API Routes
@app.post("/api/domains/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """Main endpoint: suggest domain names using multiple providers"""
    
    try:
        # Generate domain suggestions using AI
        domain_ideas = await domain_agent.generate_domain_ideas(request)
        print(f"Generated {len(domain_ideas)} domain ideas: {domain_ideas}")
        
        # Search across multiple providers
        response = await domain_agent.search_domains_parallel(domain_ideas, request)
        
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
        "version": "2.0.0",
        "providers": {
            "porkbun": bool(PORKBUN_API_KEY and PORKBUN_SECRET_KEY),
            "namecom": bool(NAMECOM_API_TOKEN and NAMECOM_USERNAME)
        }
    }

@app.get("/api/pricing")
async def get_pricing():
    """Get pricing information"""
    return {
        "status": "open_beta",
        "message": "API is currently free during development phase",
        "providers": {
            "porkbun": {
                "rate_limit": "1 domain per 10 seconds",
                "pricing": "Variable by TLD"
            },
            "namecom": {
                "rate_limit": "20 requests/sec, 3000/hour",
                "bulk_check": "Up to 50 domains per call",
                "pricing": "Variable by TLD"
            }
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