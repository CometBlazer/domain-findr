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

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
PORKBUN_API_KEY = os.getenv("PORKBUN_API_KEY", "")
PORKBUN_SECRET_KEY = os.getenv("PORKBUN_SECRET_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Redis client
try:
    redis_client = redis.from_url(REDIS_URL)
except:
    redis_client = None

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
        
        return suggestions[:request.num_choices * 3]  # Generate more than needed for filtering

    async def check_domain_availability(self, domains: List[str]) -> List[DomainResult]:
        """Check domain availability using Porkbun API"""
        results = []
        
        for domain_name in domains:
            # Add TLD if not present
            if not any(domain_name.endswith(tld) for tld in ['.com', '.net', '.org', '.io']):
                domain_name = f"{domain_name}.com"
            
            try:
                # Check cache first
                cache_key = f"domain:{domain_name}"
                if redis_client:
                    cached = redis_client.get(cache_key)
                    if cached:
                        result_data = json.loads(cached)
                        results.append(DomainResult(**result_data))
                        continue
                
                # Query Porkbun API
                async with httpx.AsyncClient() as client:
                    # First check if domain is available
                    availability_response = await client.post(
                        "https://porkbun.com/api/json/v3/domain/isAvailable",
                        json={
                            "apikey": PORKBUN_API_KEY,
                            "secretapikey": PORKBUN_SECRET_KEY,
                            "domain": domain_name
                        }
                    )
                    
                    availability_data = availability_response.json()
                    is_available = availability_data.get("status") == "SUCCESS"
                    
                    price_first_year = None
                    price_annual = None
                    pricing_details = None
                    
                    if is_available:
                        # Get pricing information
                        pricing_response = await client.post(
                            "https://porkbun.com/api/json/v3/pricing/get",
                            json={
                                "apikey": PORKBUN_API_KEY,
                                "secretapikey": PORKBUN_SECRET_KEY
                            }
                        )
                        
                        if pricing_response.status_code == 200:
                            pricing_data = pricing_response.json()
                            tld = domain_name.split('.')[-1]
                            
                            if pricing_data.get("status") == "SUCCESS" and tld in pricing_data.get("pricing", {}):
                                tld_pricing = pricing_data["pricing"][tld]
                                price_first_year = float(tld_pricing.get("registration", 0))
                                price_annual = float(tld_pricing.get("renewal", 0))
                                
                                # Check for special pricing
                                pricing_details = {
                                    "registration": price_first_year,
                                    "renewal": price_annual,
                                    "transfer": float(tld_pricing.get("transfer", 0))
                                }
                                
                                # Format deal information
                                deal_info = None
                                if price_first_year != price_annual:
                                    deal_info = f"First year: ${price_first_year}, Then: ${price_annual}/year"
                    
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
                    
                    # Cache result for 1 hour
                    if redis_client:
                        redis_client.setex(
                            cache_key, 3600, 
                            json.dumps(result.dict())
                        )
                    
                    results.append(result)
                
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

# API Routes
@app.post("/api/domains/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """Main endpoint: suggest domain names"""
    
    try:
        # Generate domain suggestions using AI
        domain_ideas = await domain_agent.generate_domain_ideas(request)
        
        # Check availability
        domain_results = await domain_agent.check_domain_availability(domain_ideas)
        
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