from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
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
import re

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Domain Finder API",
    description="AI-powered domain name suggestion API with multi-provider support",
    version="2.1.0"
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
    LY = ".ly"
    APP = ".app"
    DEV = ".dev"
    AI = ".ai"
    TECH = ".tech"
    ONLINE = ".online"
    SITE = ".site"
    WEBSITE = ".website"
    STORE = ".store"
    SHOP = ".shop"
    BIZ = ".biz"
    INFO = ".info"
    ME = ".me"
    CC = ".cc"
    TV = ".tv"
    SO = ".so"
    XYZ = ".xyz"
    CLOUD = ".cloud"
    DIGITAL = ".digital"
    AGENCY = ".agency"
    STUDIO = ".studio"
    DESIGN = ".design"
    MEDIA = ".media"
    SERVICES = ".services"
    SOLUTIONS = ".solutions"
    CONSULTING = ".consulting"
    LAB = ".lab"
    ACADEMY = ".academy"
    INSTITUTE = ".institute"
    ANY = "any"

class ProviderPreference(str, Enum):
    PORKBUN = "porkbun"
    NAMECOM = "name.com"
    ANY = "any"

class InputType(str, Enum):
    IDEA = "idea"  # Generate suggestions based on business idea/field
    EXACT_NAME = "exact_name"  # Check specific domain name(s)
    BASE_NAME = "base_name"  # Check base name with different TLDs

class DomainRequest(BaseModel):
    # Main input - can be idea, base name, or exact domain
    input_text: str = Field(..., description="Business idea, base domain name, or exact domain(s) to check")
    
    # Input type specification
    input_type: InputType = Field(default=InputType.IDEA, description="Type of input provided")
    
    # Optional fields for idea-based generation
    field: Optional[str] = Field(default="", description="Industry or field (optional for exact name/base name checks)")
    style: DomainStyle = Field(default=DomainStyle.BRANDABLE, description="Style for AI generation (ignored for exact names)")
    
    # Domain preferences
    domain_preference: DomainPreference = Field(default=DomainPreference.COM)
    provider_preference: ProviderPreference = Field(default=ProviderPreference.ANY)
    max_price: float = Field(default=50.0, description="Maximum price per year")
    num_choices: int = Field(default=5, ge=1, le=20)
    
    # Additional exact domains (for bulk checking)
    additional_domains: Optional[List[str]] = Field(default=[], description="Additional exact domains to check")

    @validator('input_text')
    def validate_input_text(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Input text cannot be empty')
        return v.strip()
    
    @validator('additional_domains')
    def validate_additional_domains(cls, v):
        if v:
            # Clean up the domains
            cleaned = []
            for domain in v:
                domain = domain.strip().lower()
                if domain and domain not in cleaned:
                    cleaned.append(domain)
            return cleaned
        return []

class DomainResult(BaseModel):
    domain: str
    available: bool
    price_first_year: Optional[float] = None
    price_annual: Optional[float] = None
    registrar: str
    deal_info: Optional[str] = None
    score: float = Field(description="AI ranking score")
    pricing_details: Optional[Dict[str, Any]] = None
    input_source: str = Field(description="How this domain was generated (ai_generated, user_provided, base_expansion)")

class DomainResponse(BaseModel):
    domains: List[DomainResult]
    request_id: str
    timestamp: datetime
    search_summary: Dict[str, Any]

# Utility functions for input parsing
class InputParser:
    @staticmethod
    def detect_input_type(input_text: str) -> InputType:
        """Auto-detect the input type based on the text"""
        input_text = input_text.strip().lower()
        
        # Check if it contains a TLD (exact domain)
        tld_pattern = r'\.(com|net|org|io|co|ly|app|dev|ai|tech|online|site|website|store|shop|biz|info|me|cc|tv|so|xyz|cloud|digital|agency|studio|design|media|services|solutions|consulting|lab|academy|institute)$'
        if re.search(tld_pattern, input_text):
            return InputType.EXACT_NAME
        
        # Check if it's a simple word/phrase without spaces (likely a base name)
        if ' ' not in input_text and len(input_text.split('.')) == 1 and len(input_text) <= 20:
            return InputType.BASE_NAME
        
        # Default to idea-based generation
        return InputType.IDEA
    
    @staticmethod
    def parse_input(request: DomainRequest) -> tuple[List[str], InputType]:
        """Parse the input and return domains to check and confirmed input type"""
        input_text = request.input_text.strip().lower()
        
        # Use provided input_type, or auto-detect if set to default
        input_type = request.input_type
        if input_type == InputType.IDEA:
            # Check if auto-detection suggests otherwise
            detected_type = InputParser.detect_input_type(input_text)
            if detected_type != InputType.IDEA:
                input_type = detected_type
        
        domains_to_check = []
        
        if input_type == InputType.EXACT_NAME:
            # Exact domain name(s) provided
            domains_to_check.append(input_text)
            # Add additional domains if provided
            domains_to_check.extend(request.additional_domains)
            
        elif input_type == InputType.BASE_NAME:
            # Base name provided - generate with different TLDs
            base_name = input_text
            if request.domain_preference == DomainPreference.ANY:
                # Generate with multiple popular TLDs
                popular_tlds = [".com", ".net", ".org", ".io", ".co", ".app", ".dev"]
                domains_to_check = [f"{base_name}{tld}" for tld in popular_tlds]
            else:
                # Use specified TLD
                tld = request.domain_preference.value
                domains_to_check = [f"{base_name}{tld}"]
            
            # Add additional domains if provided
            domains_to_check.extend(request.additional_domains)
            
        else:  # InputType.IDEA
            # Will be handled by AI generation
            domains_to_check = []
        
        return domains_to_check, input_type

# Base Domain Provider class (unchanged)
class BaseDomainProvider:
    def __init__(self, name: str):
        self.name = name
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0) -> List[DomainResult]:
        raise NotImplementedError
    
    def calculate_domain_score(self, domain: str, is_available: bool, price: Optional[float] = None, input_source: str = "ai_generated") -> float:
        """Calculate AI-based domain score"""
        score = 0.0
        
        if not is_available:
            return 0.0
        
        # Base score for user-provided domains (they know what they want)
        if input_source == "user_provided":
            score += 2.0
        elif input_source == "base_expansion":
            score += 1.0
        
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

# Porkbun Provider (updated to include input_source)
class PorkbunProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("porkbun")
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0, input_source: str = "ai_generated") -> List[DomainResult]:
        """Check domain availability using Porkbun API with rate limiting"""
        results = []
        
        # Porkbun: 1 domain per 10 seconds - very limited
        limited_domains = domains[:5] if input_source != "user_provided" else domains[:10]

        for i, domain_name in enumerate(limited_domains):
            try:
                # Check cache first
                cache_key = f"porkbun:domain:{domain_name}"
                if redis_client:
                    try:
                        cached = redis_client.get(cache_key)
                        if cached:
                            result_data = json.loads(cached)
                            result_data['input_source'] = input_source
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
                        
                        # Filter by max_price - check both first year and annual price
                        if price_first_year > max_price or price_annual > max_price:
                            continue
                        
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
                        score=self.calculate_domain_score(domain_name, is_available, price_first_year, input_source),
                        input_source=input_source
                    )
                    
                    # Cache result
                    if redis_client:
                        try:
                            expiry = 7200 if is_available else 86400
                            redis_client.setex(cache_key, expiry, json.dumps(result.dict()))
                        except Exception as e:
                            print(f"Redis write error: {e}")
                    
                    results.append(result)
                    
                    # Rate limiting: wait between requests (shorter for user-provided domains)
                    if i < len(limited_domains) - 1:
                        wait_time = 5 if input_source == "user_provided" else 10
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"Porkbun error checking domain {domain_name}: {e}")
                continue
        
        return results

# Name.com Provider (updated to include input_source)
class NameComProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("name.com")
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0, input_source: str = "ai_generated") -> List[DomainResult]:
        """Check domain availability using Name.com API"""
        results = []
        
        # Name.com allows up to 50 domains per call
        limited_domains = domains[:20] if input_source != "user_provided" else domains[:50]
        
        try:
            # Create Basic Auth header
            auth_string = f"{NAMECOM_USERNAME}:{NAMECOM_API_TOKEN}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            async with httpx.AsyncClient() as client:
                # Name.com bulk availability check
                availability_response = await client.post(
                    "https://api.name.com/v4/domains:checkAvailability",
                    json={"domainNames": limited_domains},
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
                    purchase_type = domain_info.get("purchaseType", "")
                    
                    price_first_year = None
                    price_annual = None
                    pricing_details = None
                    deal_info = None
                    
                    if is_available:
                        price_first_year = float(domain_info.get("purchasePrice", 0))
                        price_annual = float(domain_info.get("renewalPrice", 0))
                        
                        # Filter by max_price
                        if price_first_year > max_price or price_annual > max_price:
                            continue
                        
                        pricing_details = {
                            "registration": price_first_year,
                            "renewal": price_annual,
                            "premium": domain_info.get("premium", False),
                            "purchase_type": purchase_type
                        }
                        
                        if price_first_year != price_annual and price_annual > 0:
                            deal_info = f"First year: ${price_first_year:.2f}, Then: ${price_annual:.2f}/year"
                        
                        if domain_info.get("premium", False) or purchase_type in ["aftermarket_s", "aftermarket"]:
                            premium_info = f"Premium domain ({purchase_type})"
                            deal_info = f"{deal_info} - {premium_info}" if deal_info else premium_info
                    
                    result = DomainResult(
                        domain=domain_name,
                        available=is_available,
                        price_first_year=price_first_year,
                        price_annual=price_annual,
                        registrar="name.com",
                        deal_info=deal_info,
                        pricing_details=pricing_details,
                        score=self.calculate_domain_score(domain_name, is_available, price_first_year, input_source),
                        input_source=input_source
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

# Updated Domain Suggestion Agent
class DomainSuggestionAgent:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.porkbun = PorkbunProvider()
        self.namecom = NameComProvider() 
        
        # All available TLDs for ranking
        self.all_tlds = [
            ".com", ".net", ".org", ".io", ".co", ".ly", ".app", ".dev", 
            ".ai", ".tech", ".online", ".site", ".website", ".store", 
            ".shop", ".biz", ".info", ".me", ".cc", ".tv", ".so", ".xyz",
            ".cloud", ".digital", ".agency", ".studio", ".design", ".media",
            ".services", ".solutions", ".consulting", ".lab", ".academy", ".institute"
        ]
    
    async def generate_domain_ideas(self, request: DomainRequest) -> List[str]:
        """Generate domain name ideas using AI logic"""
        base_words = request.input_text.lower().split()
        field_words = request.field.lower().split() if request.field else []
        
        suggestions = []
        
        # Style-based generation
        if request.style == DomainStyle.SHORT:
            suggestions.extend([f"{word[:4]}" for word in base_words + field_words])
            suggestions.extend([f"{word1[:3]}{word2[:3]}" for word1 in base_words for word2 in field_words])
        
        elif request.style == DomainStyle.BRANDABLE:
            suffixes = ["ly", "fy", "hub", "lab", "pro", "go", "kit", "box", "co", "io"]
            prefixes = ["get", "my", "the", "smart", "quick", "easy", "auto", "super", "pro", "meta"]
            
            for word in base_words + field_words:
                suggestions.extend([f"{word}{suffix}" for suffix in suffixes])
                suggestions.extend([f"{prefix}{word}" for prefix in prefixes])
        
        elif request.style == DomainStyle.KEYWORD:
            keywords = ["online", "digital", "web", "app", "tech", "service", "cloud", "ai", "hub", "zone"]
            for word in base_words:
                suggestions.extend([f"{word}{keyword}" for keyword in keywords])
                suggestions.extend([f"{keyword}{word}" for keyword in keywords])
        
        elif request.style == DomainStyle.CREATIVE:
            variations = []
            for word in base_words + field_words:
                no_vowels = ''.join([c for c in word if c not in 'aeiou'])
                if len(no_vowels) >= 3:
                    variations.append(no_vowels)
                
                creative_endings = ["r", "d", "x", "z", "y", "ly", "fy"]
                variations.extend([f"{word}{ending}" for ending in creative_endings])
            
            suggestions.extend(variations)
        
        elif request.style == DomainStyle.PROFESSIONAL:
            prof_words = ["solutions", "services", "consulting", "group", "corp", "systems", "tech", "digital"]
            for word in base_words:
                suggestions.extend([f"{word}{prof}" for prof in prof_words])
        
        # Clean and filter suggestions
        suggestions = list(set([
            s for s in suggestions 
            if len(s) >= 3 and len(s) <= 20 and s.replace('-', '').isalnum()
        ]))
        
        return suggestions[:30]  # Generate base names without TLDs

    def generate_domain_combinations(self, base_names: List[str], request: DomainRequest) -> tuple[List[str], List[tuple]]:
        """Generate all domain combinations with TLDs and rank them"""
        
        # Determine which TLDs to use
        if request.domain_preference == DomainPreference.ANY:
            tlds_to_use = self.all_tlds
        else:
            tlds_to_use = [request.domain_preference.value]
        
        # Generate all combinations
        all_combinations = []
        for base_name in base_names:
            for tld in tlds_to_use:
                domain = f"{base_name}{tld}"
                score = self.calculate_pre_check_score(domain, request)
                all_combinations.append((domain, score))
        
        # Sort by score (highest first) and return top domains
        all_combinations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 15 domains for checking
        top_domains = [combo[0] for combo in all_combinations[:15]]
        
        return top_domains, all_combinations
    
    def calculate_pre_check_score(self, domain: str, request: DomainRequest) -> float:
        """Calculate domain score BEFORE availability check for ranking"""
        score = 0.0
        
        domain_parts = domain.split('.')
        domain_name = domain_parts[0]
        tld = '.' + domain_parts[1] if len(domain_parts) > 1 else ''
        
        # Length score (shorter is better for domain names)
        length = len(domain_name)
        if length <= 6:
            score += 3.0
        elif length <= 10:
            score += 2.0
        elif length <= 15:
            score += 1.0
        elif length <= 20:
            score += 0.5
        
        # TLD score - popular TLDs get higher scores
        tld_scores = {
            '.com': 3.0, '.io': 2.5, '.app': 2.3, '.dev': 2.2, '.ai': 2.1,
            '.net': 2.0, '.org': 1.9, '.co': 1.8, '.ly': 1.7, '.tech': 1.6,
            '.online': 1.4, '.site': 1.3, '.store': 1.2, '.shop': 1.2,
            '.me': 1.1, '.cc': 1.0, '.tv': 1.0, '.xyz': 0.8, '.biz': 0.7
        }
        score += tld_scores.get(tld, 0.5)
        
        # Readability (no numbers, easy to type)
        if domain_name.isalpha():
            score += 1.0
        
        # Brandability (good vowel-consonant balance)
        vowels = sum(1 for c in domain_name if c in 'aeiouAEIOU')
        consonants = len(domain_name) - vowels
        if vowels > 0 and consonants > 0 and vowels / len(domain_name) > 0.2:
            score += 1.0
        
        # Keyword relevance
        idea_words = request.input_text.lower().split()
        field_words = request.field.lower().split() if request.field else []
        all_keywords = idea_words + field_words
        
        for keyword in all_keywords:
            if keyword.lower() in domain_name.lower():
                score += 1.5
        
        # Avoid hyphens and numbers (reduce score)
        if '-' in domain_name or any(c.isdigit() for c in domain_name):
            score -= 1.0
        
        # Style bonus
        if request.style == DomainStyle.SHORT and length <= 6:
            score += 1.0
        elif request.style == DomainStyle.PROFESSIONAL and any(prof in domain_name for prof in ['tech', 'pro', 'corp', 'group']):
            score += 1.0
        
        return min(score, 10.0)

    async def search_domains_parallel(self, request: DomainRequest) -> DomainResponse:
        """Search domains using selected providers with new input support"""
        
        # Parse input to get domains to check
        direct_domains, actual_input_type = InputParser.parse_input(request)
        
        all_domains_to_check = []
        search_summary = {
            "input_type": actual_input_type.value,
            "original_input": request.input_text,
            "providers_used": [],
            "provider_selection": request.provider_preference.value,
            "domains_actually_checked": 0,
            "available_domains_found": 0,
            "errors": [],
            "generation_method": {}
        }
        
        if actual_input_type == InputType.IDEA:
            # Generate AI suggestions
            base_names = await self.generate_domain_ideas(request)
            top_domains, all_combinations = self.generate_domain_combinations(base_names, request)
            all_domains_to_check = top_domains
            
            search_summary["generation_method"] = {
                "type": "ai_generated",
                "base_names_generated": len(base_names),
                "total_combinations": len(all_combinations),
                "top_selected": len(top_domains)
            }
            input_source = "ai_generated"
            
        elif actual_input_type == InputType.BASE_NAME:
            # Base name expansion
            all_domains_to_check = direct_domains
            search_summary["generation_method"] = {
                "type": "base_expansion",
                "base_name": request.input_text,
                "domains_generated": len(direct_domains)
            }
            input_source = "base_expansion"
            
        else:  # InputType.EXACT_NAME
            # Exact domain checking
            all_domains_to_check = direct_domains
            search_summary["generation_method"] = {
                "type": "user_provided",
                "exact_domains": direct_domains
            }
            input_source = "user_provided"
        
        # Build provider tasks
        tasks = []
        providers_to_use = []
        
        # Porkbun
        if (request.provider_preference in [ProviderPreference.PORKBUN, ProviderPreference.ANY] and 
            PORKBUN_API_KEY and PORKBUN_SECRET_KEY):
            tasks.append(self.porkbun.check_domains(all_domains_to_check, request.domain_preference, request.max_price, input_source))
            providers_to_use.append("porkbun")
        
        # Name.com
        if (request.provider_preference in [ProviderPreference.NAMECOM, ProviderPreference.ANY] and 
            NAMECOM_API_TOKEN and NAMECOM_USERNAME):
            tasks.append(self.namecom.check_domains(all_domains_to_check, request.domain_preference, request.max_price, input_source))
            providers_to_use.append("name.com")
        
        if not tasks:
            available_providers = []
            if PORKBUN_API_KEY and PORKBUN_SECRET_KEY:
                available_providers.append("porkbun")
            if NAMECOM_API_TOKEN and NAMECOM_USERNAME:
                available_providers.append("name.com")
            
            error_msg = f"No providers available for selection '{request.provider_preference.value}'. "
            if available_providers:
                error_msg += f"Available providers: {', '.join(available_providers)}"
            else:
                error_msg += "No providers are configured with valid API credentials."
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Execute searches in parallel
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results from all providers
        all_results = []
        search_summary["providers_used"] = providers_to_use
        
        for i, results in enumerate(provider_results):
            if isinstance(results, Exception):
                provider_name = providers_to_use[i] if i < len(providers_to_use) else "unknown"
                search_summary["errors"].append(f"{provider_name}: {str(results)}")
                continue
            
            if results:
                search_summary["domains_actually_checked"] += len(results)
                available_results = [r for r in results if r.available]
                search_summary["available_domains_found"] += len(available_results)
                all_results.extend(available_results)
        
        # Remove duplicates (same domain from different providers) if using any/multiple providers
        if request.provider_preference == ProviderPreference.ANY:
            seen_domains = set()
            unique_results = []
            for result in all_results:
                if result.domain not in seen_domains:
                    seen_domains.add(result.domain)
                    unique_results.append(result)
            all_results = unique_results
        
        # Sort by score (best domains first)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Limit to requested number
        final_domains = all_results[:request.num_choices]
        
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

@app.get("/api/test-providers")
async def test_all_providers():
    """Test all available providers"""
    results = {}
    
    # Test Porkbun
    if PORKBUN_API_KEY and PORKBUN_SECRET_KEY:
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
                    results["porkbun"] = {"status": "success", "response": response.json()}
                else:
                    results["porkbun"] = {"status": "error", "message": response.text}
        except Exception as e:
            results["porkbun"] = {"status": "error", "message": str(e)}
    else:
        results["porkbun"] = {"status": "not_configured", "message": "Missing API credentials"}
    
    # Test Name.com
    if NAMECOM_API_TOKEN and NAMECOM_USERNAME:
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
                    results["namecom"] = {"status": "success", "response": response.json()}
                else:
                    results["namecom"] = {"status": "error", "message": response.text}
        except Exception as e:
            results["namecom"] = {"status": "error", "message": str(e)}
    else:
        results["namecom"] = {"status": "not_configured", "message": "Missing API credentials"}
    
    return results

# Input parsing endpoint for testing
@app.post("/api/parse-input")
async def parse_input_endpoint(request: DomainRequest):
    """Test endpoint to see how input is parsed"""
    domains_to_check, detected_type = InputParser.parse_input(request)
    
    return {
        "original_input": request.input_text,
        "provided_input_type": request.input_type.value,
        "detected_input_type": detected_type.value,
        "domains_to_check": domains_to_check,
        "additional_domains": request.additional_domains,
        "auto_detection": InputParser.detect_input_type(request.input_text).value
    }

# Main API Routes
@app.post("/api/domains/suggest", response_model=DomainResponse)
async def suggest_domains(request: DomainRequest):
    """Main endpoint: suggest domain names with support for different input types"""
    
    try:
        # Generate domain suggestions based on input type
        response = await domain_agent.search_domains_parallel(request)
        
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
        "version": "2.1.0",
        "providers": {
            "porkbun": bool(PORKBUN_API_KEY and PORKBUN_SECRET_KEY),
            "namecom": bool(NAMECOM_API_TOKEN and NAMECOM_USERNAME)
        },
        "supported_input_types": [input_type.value for input_type in InputType]
    }

@app.get("/api/pricing")
async def get_pricing():
    """Get pricing information"""
    return {
        "status": "open_beta",
        "message": "API is currently free during development phase",
        "providers": {
            "porkbun": {
                "rate_limit": "1 domain per 10 seconds (5 seconds for user-provided domains)",
                "pricing": "Variable by TLD",
                "bulk_check": "1 domain at a time"
            },
            "namecom": {
                "rate_limit": "20 requests/sec, 3000/hour",
                "bulk_check": "Up to 50 domains per call",
                "pricing": "Variable by TLD"
            }
        },
        "provider_selection": {
            "porkbun": "Use only Porkbun (slower but sometimes different pricing)",
            "name.com": "Use only Name.com (faster bulk checking)",
            "any": "Use all available providers (best coverage, removes duplicates)"
        },
        "input_types": {
            "idea": "Generate AI suggestions based on business idea and field",
            "exact_name": "Check specific domain name(s) like 'google.com'",
            "base_name": "Check base name with different TLDs like 'google' → 'google.com', 'google.io', etc."
        }
    }

@app.get("/api/examples")
async def get_examples():
    """Get example API requests for different input types"""
    return {
        "idea_based": {
            "description": "Generate AI suggestions based on business concept",
            "example": {
                "input_text": "artificial intelligence startup",
                "input_type": "idea",
                "field": "technology",
                "style": "brandable",
                "domain_preference": ".com",
                "num_choices": 5
            }
        },
        "base_name": {
            "description": "Check a base name with different TLDs",
            "example": {
                "input_text": "mycompany",
                "input_type": "base_name",
                "domain_preference": "any",  # Will check .com, .io, .net, etc.
                "num_choices": 10
            }
        },
        "exact_domain": {
            "description": "Check specific domain(s)",
            "example": {
                "input_text": "mycompany.com",
                "input_type": "exact_name",
                "additional_domains": ["mycompany.io", "mycompany.net"],
                "num_choices": 10
            }
        },
        "auto_detection": {
            "description": "Let the API auto-detect input type",
            "examples": [
                {
                    "input_text": "google.com",
                    "input_type": "idea",  # Will be auto-detected as exact_name
                    "note": "API will detect this is an exact domain"
                },
                {
                    "input_text": "myapp",
                    "input_type": "idea",  # Will be auto-detected as base_name
                    "note": "API will detect this is a base name"
                },
                {
                    "input_text": "ai powered customer service platform",
                    "input_type": "idea",  # Will stay as idea
                    "field": "technology",
                    "note": "API will detect this needs AI generation"
                }
            ]
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