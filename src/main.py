from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import httpx
import os
import json
import redis
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import logging
from dotenv import load_dotenv
import base64
import re
import uuid
import time
from collections import defaultdict
import threading

# Load environment variables from .env file
load_dotenv()

# ==================== CONFIGURATION ====================
# Rate limiting configuration (requests per minute)
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

# All available domain extensions - edit this list to add/remove TLDs
AVAILABLE_TLDS = [
    # Popular TLDs (Tier 1)
    ".com", ".net", ".org", ".io", ".co", ".ai", ".app", ".dev",
    
    # Business TLDs (Tier 2)
    ".biz", ".pro", ".inc", ".corp", ".ltd", ".company", ".business",
    
    # Tech TLDs (Tier 3)
    ".tech", ".digital", ".cloud", ".online", ".website", ".site",
    
    # Creative TLDs (Tier 4)
    ".ly", ".me", ".cc", ".tv", ".so", ".sh", ".xyz", ".fun",
    
    # Industry-specific TLDs (Tier 5)
    ".store", ".shop", ".agency", ".studio", ".design", ".media",
    ".services", ".solutions", ".consulting", ".lab", ".academy",
    ".institute", ".guide", ".fit", ".life"
]

# Common English words for domain ranking (add more as needed)
COMMON_WORDS = {
    'get', 'my', 'the', 'app', 'web', 'go', 'pro', 'hub', 'lab', 'box',
    'tech', 'ai', 'smart', 'quick', 'easy', 'fast', 'auto', 'super',
    'digital', 'online', 'cloud', 'data', 'system', 'service', 'solution',
    'platform', 'network', 'secure', 'global', 'local', 'mobile', 'social',
    'business', 'company', 'group', 'team', 'work', 'studio', 'agency',
    'design', 'creative', 'media', 'content', 'marketing', 'sales',
    'finance', 'health', 'education', 'learning', 'training', 'consulting',
    'expert', 'master', 'elite', 'premium', 'plus', 'max', 'ultra',
    'best', 'top', 'first', 'new', 'next', 'future', 'modern', 'advanced'
}

# Word quality scoring for better domain ranking
WORD_QUALITY_SCORES = {
    # High-value business words
    'app': 3.0, 'tech': 3.0, 'pro': 3.0, 'hub': 3.0, 'lab': 3.0,
    'cloud': 2.8, 'ai': 2.8, 'digital': 2.8, 'smart': 2.8,
    'solution': 2.5, 'platform': 2.5, 'system': 2.5, 'network': 2.5,
    
    # Medium-value words
    'web': 2.0, 'online': 2.0, 'service': 2.0, 'data': 2.0,
    'business': 1.8, 'company': 1.8, 'group': 1.8, 'team': 1.8,
    
    # Lower-value but still useful
    'get': 1.0, 'my': 1.0, 'the': 0.5, 'go': 1.2, 'box': 1.5
}

# ==================== RATE LIMITING ====================
class RateLimiter:
    def __init__(self, max_requests: int, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)"""
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.window_seconds
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(current_time)
                return True, 0
            else:
                # Calculate retry after time
                oldest_request = min(self.requests[client_id])
                retry_after = int(oldest_request + self.window_seconds - current_time) + 1
                return False, retry_after

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS_PER_MINUTE)

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Use X-Forwarded-For if behind proxy, otherwise use client IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

async def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting"""
    client_id = get_client_id(request)
    allowed, retry_after = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute allowed",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )

# Initialize FastAPI app
app = FastAPI(
    title="Domains API",
    description="Domain name suggestion API with multi-provider support and intelligent ranking",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

print(f"Rate Limit: {RATE_LIMIT_REQUESTS_PER_MINUTE} requests/minute")
print(f"Available TLDs: {len(AVAILABLE_TLDS)} extensions loaded")
print(f"Loaded Porkbun API Key: {'✅ Yes' if PORKBUN_API_KEY else '❌ No'}")
print(f"Loaded Porkbun Secret: {'✅ Yes' if PORKBUN_SECRET_KEY else '❌ No'}")
print(f"Loaded Name.com Token: {'✅ Yes' if NAMECOM_API_TOKEN else '❌ No'}")
print(f"Loaded Name.com Username: {'✅ Yes' if NAMECOM_USERNAME else '❌ No'}")

# Redis client with better error handling
def get_redis_client():
    try:
        redis_url = REDIS_URL

        if redis_url.startswith('rediss://'):
            client = redis.from_url(
                redis_url,
                ssl_cert_reqs=None,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
        else:
            client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )

        client.ping()
        print(f"✅ Redis connected successfully to: {redis_url.split('@')[1] if '@' in redis_url else 'localhost'}")
        return client

    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("🔄 Running without caching (Redis unavailable)")
        return None
    except redis.AuthenticationError as e:
        print(f"❌ Redis authentication failed: {e}")
        print("🔧 Check your REDIS_URL and token")
        return None
    except Exception as e:
        print(f"❌ Redis error: {e} - running without caching")
        return None

# Cache utility functions
def safe_cache_set(redis_client, key: str, value: dict, expiry: int):
    if not redis_client:
        return False
    try:
        redis_client.setex(key, expiry, json.dumps(value, default=str))
        return True
    except Exception as e:
        print(f"Cache write error for key {key}: {e}")
        return False

def safe_cache_get(redis_client, key: str):
    if not redis_client:
        return None
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        print(f"Cache read error for key {key}: {e}")
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
    SH = ".sh"
    GUIDE = ".guide"
    INC = ".inc"
    FIT = ".fit"
    LIFE = ".life"
    PRO = ".pro"
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
    
    # Domain preferences - CHANGED DEFAULT TO NAME.COM
    domain_preference: DomainPreference = Field(default=DomainPreference.COM)
    provider_preference: ProviderPreference = Field(default=ProviderPreference.NAMECOM, description="Default is name.com due to better rate limits")
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
    score: float = Field(description="Domain quality ranking score")
    pricing_details: Optional[Dict[str, Any]] = None
    input_source: str = Field(description="How this domain was generated (ai_generated, user_provided, base_expansion)")
    ranking_factors: Optional[Dict[str, Any]] = Field(description="Factors that contributed to the ranking score")

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
        tld_pattern = r'\.(com|net|org|io|co|ly|app|dev|ai|tech|online|site|website|store|shop|biz|info|me|cc|tv|so|xyz|cloud|digital|agency|studio|design|media|services|solutions|consulting|lab|academy|institute|sh|guide|inc|fit|life|pro)$'
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
                # Use available TLDs from configuration
                domains_to_check = [f"{base_name}{tld}" for tld in AVAILABLE_TLDS[:15]]  # Limit to first 15 for performance
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

# IMPROVED Domain Ranking System
class DomainRanker:
    @staticmethod
    def calculate_domain_score(domain: str, is_available: bool, price: Optional[float] = None, 
                             input_source: str = "ai_generated", original_keywords: List[str] = None) -> tuple[float, Dict[str, Any]]:
        """
        Calculate GoDaddy-style domain quality score with detailed ranking factors
        Returns (score, ranking_factors_dict)
        """
        if not is_available:
            return 0.0, {"available": False}
        
        domain_parts = domain.split('.')
        domain_name = domain_parts[0]
        tld = '.' + domain_parts[1] if len(domain_parts) > 1 else ''
        
        ranking_factors = {}
        score = 0.0
        
        # 1. INPUT SOURCE BONUS (30% weight)
        source_scores = {
            "user_provided": 8.0,    # User knows what they want
            "base_expansion": 6.0,   # Based on user's base name
            "ai_generated": 4.0      # AI suggestion
        }
        source_score = source_scores.get(input_source, 4.0)
        score += source_score
        ranking_factors["source_bonus"] = source_score
        
        # 2. LENGTH SCORE (20% weight) - Shorter is generally better
        length = len(domain_name)
        if length <= 5:
            length_score = 8.0
        elif length <= 8:
            length_score = 7.0
        elif length <= 12:
            length_score = 5.0
        elif length <= 16:
            length_score = 3.0
        else:
            length_score = 1.0
        
        score += length_score
        ranking_factors["length_score"] = length_score
        ranking_factors["domain_length"] = length
        
        # 3. TLD QUALITY (15% weight) - Premium TLDs score higher
        tld_scores = {
            '.com': 10.0, '.io': 8.0, '.ai': 7.5, '.app': 7.0, '.dev': 6.5,
            '.net': 6.0, '.org': 6.0, '.co': 5.5, '.tech': 5.0, '.pro': 5.0,
            '.ly': 4.5, '.me': 4.0, '.cc': 3.5, '.tv': 3.5, '.online': 3.0,
            '.site': 2.5, '.store': 4.0, '.shop': 4.0, '.biz': 2.0, '.info': 1.5
        }
        tld_score = tld_scores.get(tld, 2.0)
        score += tld_score
        ranking_factors["tld_score"] = tld_score
        ranking_factors["tld"] = tld
        
        # 4. WORD QUALITY (25% weight) - Real words vs gibberish
        word_score = DomainRanker._calculate_word_quality(domain_name, original_keywords or [])
        score += word_score
        ranking_factors["word_quality"] = word_score
        
        # 5. MEMORABILITY (10% weight) - Easy to remember and type
        memorability_score = DomainRanker._calculate_memorability(domain_name)
        score += memorability_score
        ranking_factors["memorability"] = memorability_score
        
        # 6. BRANDABILITY (10% weight) - Sounds like a brand
        brandability_score = DomainRanker._calculate_brandability(domain_name)
        score += brandability_score
        ranking_factors["brandability"] = brandability_score
        
        # 7. PRICE FACTOR (5% weight) - Lower price is better
        if price is not None:
            if price <= 15:
                price_score = 3.0
            elif price <= 25:
                price_score = 2.0
            elif price <= 50:
                price_score = 1.0
            else:
                price_score = 0.5
        else:
            price_score = 1.5
        
        score += price_score
        ranking_factors["price_score"] = price_score
        ranking_factors["price"] = price
        
        # 8. KEYWORD RELEVANCE (15% weight) - Contains relevant keywords
        if original_keywords:
            keyword_score = DomainRanker._calculate_keyword_relevance(domain_name, original_keywords)
            score += keyword_score
            ranking_factors["keyword_relevance"] = keyword_score
        
        # Normalize score to 0-10 range
        final_score = min(score / 8.0, 10.0)
        ranking_factors["final_score"] = final_score
        
        return final_score, ranking_factors
    
    @staticmethod
    def _calculate_word_quality(domain_name: str, keywords: List[str]) -> float:
        """Calculate word quality score based on real words vs gibberish"""
        score = 0.0
        
        # Check if domain contains complete common words
        domain_lower = domain_name.lower()
        
        # High bonus for containing original keywords
        for keyword in keywords:
            if keyword.lower() in domain_lower:
                score += 4.0
                break
        
        # Check for common English words
        words_found = 0
        for word in COMMON_WORDS:
            if word in domain_lower:
                word_quality = WORD_QUALITY_SCORES.get(word, 1.0)
                score += word_quality
                words_found += 1
        
        # Bonus for multiple recognizable words
        if words_found >= 2:
            score += 2.0
        elif words_found == 1:
            score += 1.0
        
        # Penalty for numbers and hyphens
        if any(c.isdigit() for c in domain_name):
            score -= 2.0
        if '-' in domain_name:
            score -= 1.0
        if '_' in domain_name:
            score -= 3.0  # Underscores are bad for domains
        
        # Check if it's pronounceable (vowel-consonant balance)
        vowels = sum(1 for c in domain_name if c.lower() in 'aeiou')
        consonants = len(domain_name) - vowels
        
        if vowels == 0 or consonants == 0:
            score -= 2.0  # All vowels or all consonants
        elif vowels / len(domain_name) > 0.6:
            score -= 1.0  # Too many vowels
        elif vowels / len(domain_name) < 0.15:
            score -= 1.0  # Too few vowels
        else:
            score += 1.0   # Good balance
        
        return max(score, 0.0)
    
    @staticmethod
    def _calculate_memorability(domain_name: str) -> float:
        """Calculate how memorable/easy to type the domain is"""
        score = 2.0
        
        # Easy to type - no complex letter combinations
        difficult_combos = ['qx', 'qz', 'xz', 'vw', 'jk', 'pq']
        for combo in difficult_combos:
            if combo in domain_name.lower():
                score -= 0.5
        
        # Repeated letters can be hard to remember
        for i in range(len(domain_name) - 2):
            if domain_name[i] == domain_name[i+1] == domain_name[i+2]:
                score -= 1.0  # Three or more repeated letters
                break
        
        # Simple patterns are memorable
        if domain_name.lower() == domain_name.lower()[::-1]:  # Palindrome
            score += 1.0
        
        # Alternating vowel-consonant pattern is good
        pattern_score = 0
        for i in range(len(domain_name) - 1):
            c1_vowel = domain_name[i].lower() in 'aeiou'
            c2_vowel = domain_name[i+1].lower() in 'aeiou'
            if c1_vowel != c2_vowel:
                pattern_score += 0.1
        
        score += min(pattern_score, 1.0)
        
        return max(score, 0.0)
    
    @staticmethod
    def _calculate_brandability(domain_name: str) -> float:
        """Calculate how brandable the domain sounds"""
        score = 1.0
        
        # Ends with brandable suffixes
        brandable_suffixes = ['ly', 'fy', 'hub', 'lab', 'pro', 'go', 'kit', 'box']
        for suffix in brandable_suffixes:
            if domain_name.lower().endswith(suffix):
                score += 1.5
                break
        
        # Starts with brandable prefixes
        brandable_prefixes = ['get', 'my', 'smart', 'quick', 'easy', 'auto', 'super', 'pro', 'meta']
        for prefix in brandable_prefixes:
            if domain_name.lower().startswith(prefix):
                score += 1.0
                break
        
        # Mixed case sensitivity (CamelCase is brandable)
        if any(c.isupper() for c in domain_name) and any(c.islower() for c in domain_name):
            score += 0.5
        
        # Unique character combinations
        unique_chars = len(set(domain_name.lower()))
        if unique_chars / len(domain_name) > 0.7:  # High character diversity
            score += 0.5
        
        return max(score, 0.0)
    
    @staticmethod
    def _calculate_keyword_relevance(domain_name: str, keywords: List[str]) -> float:
        """Calculate relevance to original search keywords"""
        if not keywords:
            return 0.0
        
        score = 0.0
        domain_lower = domain_name.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            if keyword_lower == domain_lower:
                score += 5.0
            # Contains entire keyword
            elif keyword_lower in domain_lower:
                score += 3.0
            # Contains parts of keyword
            elif len(keyword_lower) > 3:
                for i in range(len(keyword_lower) - 2):
                    if keyword_lower[i:i+3] in domain_lower:
                        score += 0.5
        
        return min(score, 5.0)

# Base Domain Provider class (updated with new ranking)
class BaseDomainProvider:
    def __init__(self, name: str):
        self.name = name
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0, 
                          input_source: str = "ai_generated", original_keywords: List[str] = None) -> List[DomainResult]:
        raise NotImplementedError

# Porkbun Provider (updated with new ranking)
class PorkbunProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("porkbun")
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0, 
                          input_source: str = "ai_generated", original_keywords: List[str] = None) -> List[DomainResult]:
        """Check domain availability using Porkbun API with rate limiting"""
        results = []
        
        # Porkbun: 1 domain per 10 seconds - very limited
        limited_domains = domains[:5] if input_source != "user_provided" else domains[:10]

        for i, domain_name in enumerate(limited_domains):
            try:
                # Check cache first
                cache_key = f"porkbun:domain:{domain_name}"
                cached_data = safe_cache_get(redis_client, cache_key)
                if cached_data:
                    cached_data['input_source'] = input_source
                    # Recalculate score with new keywords
                    score, ranking_factors = DomainRanker.calculate_domain_score(
                        domain_name, cached_data['available'], 
                        cached_data.get('price_first_year'), input_source, original_keywords
                    )
                    cached_data['score'] = score
                    cached_data['ranking_factors'] = ranking_factors
                    results.append(DomainResult(**cached_data))
                    continue
                
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
                    
                    # Calculate improved score
                    score, ranking_factors = DomainRanker.calculate_domain_score(
                        domain_name, is_available, price_first_year, input_source, original_keywords
                    )
                    
                    result = DomainResult(
                        domain=domain_name,
                        available=is_available,
                        price_first_year=price_first_year,
                        price_annual=price_annual,
                        registrar="porkbun",
                        deal_info=deal_info,
                        pricing_details=pricing_details,
                        score=score,
                        input_source=input_source,
                        ranking_factors=ranking_factors
                    )
                    
                    # Cache result
                    expiry = 7200 if is_available else 86400
                    safe_cache_set(redis_client, cache_key, result.dict(), expiry)
                    
                    results.append(result)
                    
                    # Rate limiting: wait between requests (shorter for user-provided domains)
                    if i < len(limited_domains) - 1:
                        wait_time = 5 if input_source == "user_provided" else 10
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"Porkbun error checking domain {domain_name}: {e}")
                continue
        
        return results

# Name.com Provider (updated with new ranking)
class NameComProvider(BaseDomainProvider):
    def __init__(self):
        super().__init__("name.com")
    
    async def check_domains(self, domains: List[str], tld_preference: str, max_price: float = 50.0, 
                          input_source: str = "ai_generated", original_keywords: List[str] = None) -> List[DomainResult]:
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
                    
                    # Calculate improved score
                    score, ranking_factors = DomainRanker.calculate_domain_score(
                        domain_name, is_available, price_first_year, input_source, original_keywords
                    )
                    
                    result = DomainResult(
                        domain=domain_name,
                        available=is_available,
                        price_first_year=price_first_year,
                        price_annual=price_annual,
                        registrar="name.com",
                        deal_info=deal_info,
                        pricing_details=pricing_details,
                        score=score,
                        input_source=input_source,
                        ranking_factors=ranking_factors
                    )
                    
                    # Cache result
                    cache_key = f"namecom:domain:{domain_name}"
                    expiry = 7200 if is_available else 86400
                    safe_cache_set(redis_client, cache_key, result.dict(), expiry)
                    
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
    
    async def generate_domain_ideas(self, request: DomainRequest) -> List[str]:
        """Generate domain name ideas using improved logic"""
        base_words = request.input_text.lower().split()
        field_words = request.field.lower().split() if request.field else []
        all_keywords = base_words + field_words
        
        suggestions = []
        
        # Style-based generation with better quality
        if request.style == DomainStyle.SHORT:
            # Create meaningful short combinations
            for word in base_words + field_words:
                if len(word) > 4:
                    suggestions.append(word[:4])
                    suggestions.append(word[:5])
            
            # Combine short versions of words
            for i, word1 in enumerate(base_words):
                for word2 in field_words:
                    if i < 3:  # Limit combinations
                        suggestions.append(f"{word1[:3]}{word2[:3]}")
        
        elif request.style == DomainStyle.BRANDABLE:
            # Use high-quality brandable patterns
            high_quality_suffixes = ["ly", "fy", "hub", "lab", "pro", "go", "kit", "app"]
            high_quality_prefixes = ["get", "my", "smart", "quick", "easy", "super", "pro"]
            
            for word in base_words + field_words:
                for suffix in high_quality_suffixes:
                    suggestions.append(f"{word}{suffix}")
                for prefix in high_quality_prefixes:
                    suggestions.append(f"{prefix}{word}")
        
        elif request.style == DomainStyle.KEYWORD:
            # Use relevant industry keywords
            tech_keywords = ["app", "tech", "digital", "cloud", "ai", "smart", "hub", "lab"]
            business_keywords = ["pro", "solutions", "services", "group", "corp", "global"]
            
            keywords_to_use = tech_keywords if any(tech in request.field.lower() for tech in ["tech", "software", "app", "digital"]) else business_keywords
            
            for word in base_words:
                for keyword in keywords_to_use[:5]:  # Limit to best keywords
                    suggestions.append(f"{word}{keyword}")
                    suggestions.append(f"{keyword}{word}")
        
        elif request.style == DomainStyle.CREATIVE:
            # More controlled creative variations
            for word in base_words + field_words:
                if len(word) > 4:
                    # Smart vowel removal (keep pronounceable)
                    consonant_version = ''.join([c for i, c in enumerate(word) if c not in 'aeiou' or i == 0])
                    if len(consonant_version) >= 3:
                        suggestions.append(consonant_version)
                
                # Modern creative endings
                modern_endings = ["r", "ly", "fy", "x"]
                for ending in modern_endings:
                    suggestions.append(f"{word}{ending}")
        
        elif request.style == DomainStyle.PROFESSIONAL:
            # Professional business terms
            prof_terms = ["solutions", "consulting", "group", "systems", "technologies", "enterprises"]
            for word in base_words:
                for term in prof_terms:
                    suggestions.append(f"{word}{term}")
        
        # Clean and filter suggestions with quality check
        clean_suggestions = []
        for s in suggestions:
            if (len(s) >= 3 and len(s) <= 20 and 
                s.replace('-', '').isalnum() and
                not s.startswith('-') and not s.endswith('-')):
                clean_suggestions.append(s)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for item in clean_suggestions:
            if item not in seen:
                seen.add(item)
                unique_suggestions.append(item)
        
        return unique_suggestions[:30]  # Return top 30 base names

    def generate_domain_combinations(self, base_names: List[str], request: DomainRequest) -> tuple[List[str], List[tuple]]:
        """Generate all domain combinations with TLDs and rank them"""
        
        # Determine which TLDs to use based on preference
        if request.domain_preference == DomainPreference.ANY:
            tlds_to_use = AVAILABLE_TLDS[:15]  # Use first 15 TLDs for performance
        else:
            tlds_to_use = [request.domain_preference.value]
        
        # Generate all combinations with pre-scoring
        all_combinations = []
        keywords = request.input_text.lower().split() + (request.field.lower().split() if request.field else [])
        
        for base_name in base_names:
            for tld in tlds_to_use:
                domain = f"{base_name}{tld}"
                # Pre-calculate score for ranking (assume available for now)
                score, _ = DomainRanker.calculate_domain_score(domain, True, None, "ai_generated", keywords)
                all_combinations.append((domain, score))
        
        # Sort by score (highest first)
        all_combinations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top domains for checking
        top_domains = [combo[0] for combo in all_combinations[:20]]
        
        return top_domains, all_combinations

    async def search_domains_parallel(self, request: DomainRequest) -> DomainResponse:
        """Search domains using selected providers with improved ranking"""

        direct_domains, actual_input_type = InputParser.parse_input(request)
        original_keywords = request.input_text.lower().split() + (request.field.lower().split() if request.field else [])

        all_domains_to_check = []
        search_summary = {
            "input_type": actual_input_type.value,
            "original_input": request.input_text,
            "providers_used": [],
            "provider_selection": request.provider_preference.value,
            "domains_generated_for_check": 0,
            "domains_actually_checked_by_providers": 0,
            "available_domains_found": 0,
            "errors": [],
            "generation_method": {}
        }

        if actual_input_type == InputType.IDEA:
            base_names = await self.generate_domain_ideas(request)
            top_domains, all_combinations = self.generate_domain_combinations(base_names, request)
            all_domains_to_check = top_domains
            search_summary["generation_method"] = {
                "type": "ai_generated",
                "base_names_generated": len(base_names),
                "total_combinations": len(all_combinations),
                "top_selected_for_check": len(top_domains)
            }
            input_source = "ai_generated"
        elif actual_input_type == InputType.BASE_NAME:
            all_domains_to_check = direct_domains
            search_summary["generation_method"] = {
                "type": "base_expansion",
                "base_name": request.input_text,
                "domains_generated": len(direct_domains)
            }
            input_source = "base_expansion"
        else:  # InputType.EXACT_NAME
            all_domains_to_check = direct_domains
            search_summary["generation_method"] = {
                "type": "user_provided",
                "exact_domains": direct_domains
            }
            input_source = "user_provided"

        search_summary["domains_generated_for_check"] = len(all_domains_to_check)

        # Prepare provider tasks
        tasks = []
        providers_to_use = []
        
        if (request.provider_preference in [ProviderPreference.PORKBUN, ProviderPreference.ANY] and
            PORKBUN_API_KEY and PORKBUN_SECRET_KEY):
            tasks.append(self.porkbun.check_domains(all_domains_to_check, request.domain_preference.value, 
                                                  request.max_price, input_source, original_keywords))
            providers_to_use.append("porkbun")
            
        if (request.provider_preference in [ProviderPreference.NAMECOM, ProviderPreference.ANY] and
            NAMECOM_API_TOKEN and NAMECOM_USERNAME):
            tasks.append(self.namecom.check_domains(all_domains_to_check, request.domain_preference.value, 
                                                   request.max_price, input_source, original_keywords))
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

        # Execute provider checks
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        search_summary["providers_used"] = providers_to_use

        # Combine, deduplicate, and sort results
        unique_results: Dict[str, DomainResult] = {}
        for i, results in enumerate(provider_results):
            if isinstance(results, Exception):
                provider_name = providers_to_use[i] if i < len(providers_to_use) else "unknown"
                search_summary["errors"].append(f"{provider_name}: {str(results)}")
                continue

            if results:
                search_summary["domains_actually_checked_by_providers"] += len(results)
                for r in results:
                    # Only add if available and not already present (first provider wins for pricing)
                    if r.available and r.domain not in unique_results:
                        unique_results[r.domain] = r
        
        # Sort by improved score (highest first)
        sorted_domains = sorted(
            unique_results.values(), 
            key=lambda d: d.score, 
            reverse=True
        )

        # Limit results to requested number
        final_domains = sorted_domains[:request.num_choices]
        search_summary["available_domains_found"] = len(final_domains)

        return DomainResponse(
            domains=final_domains,
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            search_summary=search_summary
        )

# Initialize AI agent
domain_agent = DomainSuggestionAgent()

# ==================== API ENDPOINTS ====================

# Test endpoints
@app.get("/api/test-porkbun")
async def test_porkbun_connection(request: Request, _: None = Depends(rate_limit_dependency)):
    """
    Test Porkbun API connection
    
    **Rate Limited**: Counts against your rate limit  
    **Authentication**: API keys from environment variables
    """
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
async def test_namecom_connection(request: Request, _: None = Depends(rate_limit_dependency)):
    """
    Test Name.com API connection
    
    **Rate Limited**: Counts against your rate limit  
    **Authentication**: API credentials from environment variables
    """
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
async def test_all_providers(request: Request, _: None = Depends(rate_limit_dependency)):
    """
    Test all configured domain providers
    
    **Rate Limited**: Counts against your rate limit  
    **Returns**: Status of each provider (success/error/not_configured)
    """
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

@app.post("/api/parse-input")
async def parse_input_endpoint(request_data: DomainRequest, request: Request, _: None = Depends(rate_limit_dependency)):
    """
    Test endpoint to see how input text is parsed and categorized
    
    **Rate Limited**: Counts against your rate limit  
    **Use Case**: Debug how the API interprets your input before making domain suggestions
    """
    domains_to_check, detected_type = InputParser.parse_input(request_data)
    
    return {
        "original_input": request_data.input_text,
        "provided_input_type": request_data.input_type.value,
        "detected_input_type": detected_type.value,
        "domains_to_check": domains_to_check,
        "additional_domains": request_data.additional_domains,
        "auto_detection": InputParser.detect_input_type(request_data.input_text).value
    }

# Main API Routes
@app.post("/api/domains/suggest", response_model=DomainResponse)
async def suggest_domains(request_data: DomainRequest, request: Request, _: None = Depends(rate_limit_dependency)):
    """
    **Main Endpoint**: Generate domain name suggestions with intelligent ranking
    
    **Rate Limited**: {RATE_LIMIT_REQUESTS_PER_MINUTE}/minute  
    **Default Provider**: name.com (due to better rate limits than Porkbun)
    
    ## Input Types:
    - **idea**: Generate AI suggestions based on business concept
    - **exact_name**: Check specific domain(s) like 'google.com'  
    - **base_name**: Check base name with different TLDs like 'google'
    
    ## Domain Ranking:
    Uses GoDaddy-style intelligent ranking based on:
    - Word quality (real words vs gibberish)
    - Memorability and brandability
    - Length and TLD quality
    - Keyword relevance
    - Price factors
    
    ## Provider Notes:
    - **name.com**: Fast bulk checking (default, recommended)
    - **porkbun**: Slower due to strict rate limits (1 domain/10 seconds)
    - **any**: Uses all available providers for best coverage
    """
    
    try:
        response = await domain_agent.search_domains_parallel(request_data)
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating domain suggestions: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint - **NOT rate limited**
    
    **Returns**: API status, version, and provider availability
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.2.0",
        "rate_limit": f"{RATE_LIMIT_REQUESTS_PER_MINUTE} requests/minute",
        "available_tlds": len(AVAILABLE_TLDS),
        "providers": {
            "porkbun": bool(PORKBUN_API_KEY and PORKBUN_SECRET_KEY),
            "namecom": bool(NAMECOM_API_TOKEN and NAMECOM_USERNAME)
        },
        "supported_input_types": [input_type.value for input_type in InputType],
        "default_provider": "name.com"
    }

@app.get("/api/pricing")
async def get_pricing():
    """
    Get pricing information and API limits - **NOT rate limited**
    
    **Returns**: Current pricing, rate limits, and provider comparison
    """
    return {
        "status": "open_beta",
        "message": "API is currently free during development phase",
        "rate_limiting": {
            "requests_per_minute": RATE_LIMIT_REQUESTS_PER_MINUTE,
            "identification": "by IP address",
            "retry_after": "provided in error response"
        },
        "providers": {
            "namecom": {
                "rate_limit": "20 requests/sec, 3000/hour (API limit)",
                "bulk_check": "Up to 50 domains per call",
                "pricing": "Variable by TLD",
                "recommended": "✅ Default choice - fast and reliable"
            },
            "porkbun": {
                "rate_limit": "1 domain per 10 seconds (very slow)",
                "pricing": "Variable by TLD, sometimes different from Name.com",
                "bulk_check": "1 domain at a time",
                "recommended": "⚠️  Only for individual domain checks"
            }
        },
        "provider_selection": {
            "name.com": "Use only Name.com (fast, recommended)",
            "porkbun": "Use only Porkbun (very slow, different pricing)",
            "any": "Use all available providers (removes duplicates, slower)"
        },
        "input_types": {
            "idea": "Generate AI suggestions: 'ai customer service platform'",
            "exact_name": "Check specific domains: 'mycompany.com'",
            "base_name": "Check base with TLDs: 'mycompany' → .com, .io, .ai, etc."
        }
    }

@app.get("/api/examples")
async def get_examples():
    """
    Get example API requests for different use cases - **NOT rate limited**
    
    **Returns**: Complete example requests for all input types
    """
    return {
        "idea_based_generation": {
            "description": "Generate AI suggestions based on business concept",
            "example_request": {
                "input_text": "artificial intelligence customer service platform",
                "input_type": "idea",
                "field": "technology",
                "style": "brandable",
                "domain_preference": ".com",
                "provider_preference": "name.com",
                "max_price": 50.0,
                "num_choices": 5
            },
            "expected_suggestions": ["aiservicehub.com", "smartcustomerpro.com", "serviceailab.com"]
        },
        "base_name_expansion": {
            "description": "Check a base name with different TLDs",
            "example_request": {
                "input_text": "mycompany",
                "input_type": "base_name",
                "domain_preference": "any",
                "provider_preference": "name.com",
                "num_choices": 10
            },
            "will_check": ["mycompany.com", "mycompany.io", "mycompany.ai", "mycompany.app", "..."]
        },
        "exact_domain_check": {
            "description": "Check specific domain(s) for availability",
            "example_request": {
                "input_text": "mycompany.com",
                "input_type": "exact_name",
                "additional_domains": ["mycompany.io", "mycompany.net", "myapp.ai"],
                "provider_preference": "name.com",
                "num_choices": 10
            },
            "will_check": ["mycompany.com", "mycompany.io", "mycompany.net", "myapp.ai"]
        },
        "auto_detection": {
            "description": "Let the API automatically detect input type",
            "examples": [
                {
                    "input": "google.com",
                    "detected_as": "exact_name",
                    "note": "Contains TLD, treated as exact domain"
                },
                {
                    "input": "myapp", 
                    "detected_as": "base_name",
                    "note": "Single word, treated as base name for TLD expansion"
                },
                {
                    "input": "ai powered customer service",
                    "detected_as": "idea", 
                    "note": "Multiple words, needs AI generation"
                }
            ]
        },
        "advanced_options": {
            "description": "Advanced configuration options",
            "example_request": {
                "input_text": "fintech startup",
                "input_type": "idea",
                "field": "financial technology",
                "style": "professional",
                "domain_preference": ".com",
                "provider_preference": "any",
                "max_price": 25.0,
                "num_choices": 15
            },
            "style_options": {
                "short": "Generate short, concise domains (4-6 chars)",
                "brandable": "Modern, memorable brand names (recommended)",
                "keyword": "Include relevant industry keywords",
                "creative": "Unique, creative variations",
                "professional": "Business-focused, corporate style"
            }
        }
    }

@app.get("/api/tlds")
async def get_available_tlds():
    """
    Get list of all supported Top Level Domains (TLDs) - **NOT rate limited**
    
    **Returns**: Complete list of available domain extensions with tier information
    """
    return {
        "total_tlds": len(AVAILABLE_TLDS),
        "tlds": AVAILABLE_TLDS,
        "tld_tiers": {
            "tier_1_premium": [".com", ".net", ".org", ".io", ".co", ".ai", ".app", ".dev"],
            "tier_2_business": [".biz", ".pro", ".inc", ".corp", ".ltd", ".company", ".business"],
            "tier_3_tech": [".tech", ".digital", ".cloud", ".online", ".website", ".site"],
            "tier_4_creative": [".ly", ".me", ".cc", ".tv", ".so", ".sh", ".xyz", ".fun"],
            "tier_5_industry": [".store", ".shop", ".agency", ".studio", ".design", ".media", ".services", ".solutions", ".consulting", ".lab", ".academy", ".institute", ".guide", ".fit", ".life"]
        },
        "ranking_info": {
            "highest_scored": ".com gets 10.0 points",
            "high_value": ".io (8.0), .ai (7.5), .app (7.0)",
            "medium_value": ".net, .org (6.0), .co (5.5)",
            "specialty": ".store, .shop (4.0 for e-commerce)",
            "note": "TLD score is 15% of total domain ranking"
        }
    }

@app.get("/api/ranking")
async def get_ranking_info():
    """
    Get detailed information about domain ranking algorithm - **NOT rate limited**
    
    **Returns**: Complete breakdown of how domains are scored and ranked
    """
    return {
        "ranking_system": "GoDaddy-style intelligent domain ranking",
        "total_score_range": "0.0 to 10.0",
        "ranking_factors": {
            "source_bonus": {
                "weight": "30%",
                "description": "How the domain was generated",
                "scores": {
                    "user_provided": "8.0 - User specified exact domain",
                    "base_expansion": "6.0 - Based on user's base name", 
                    "ai_generated": "4.0 - AI suggestion"
                }
            },
            "length_score": {
                "weight": "20%",
                "description": "Domain name length (shorter is better)",
                "scores": {
                    "≤5 chars": "8.0",
                    "6-8 chars": "7.0", 
                    "9-12 chars": "5.0",
                    "13-16 chars": "3.0",
                    ">16 chars": "1.0"
                }
            },
            "tld_quality": {
                "weight": "15%",
                "description": "Top Level Domain quality",
                "top_scores": {
                    ".com": "10.0",
                    ".io": "8.0",
                    ".ai": "7.5",
                    ".app": "7.0",
                    ".dev": "6.5"
                }
            },
            "word_quality": {
                "weight": "25%",
                "description": "Real words vs gibberish, pronounceability",
                "factors": [
                    "Contains original keywords (+4.0)",
                    "Contains common English words (variable bonus)",
                    "Good vowel-consonant balance (+1.0)",
                    "Numbers penalty (-2.0)",
                    "Hyphens penalty (-1.0)",
                    "Underscores penalty (-3.0)"
                ]
            },
            "memorability": {
                "weight": "10%", 
                "description": "Easy to remember and type",
                "factors": [
                    "Avoids difficult letter combinations",
                    "No excessive repeated letters",
                    "Good alternating vowel-consonant pattern",
                    "Palindromes get bonus (+1.0)"
                ]
            },
            "brandability": {
                "weight": "10%",
                "description": "Sounds like a professional brand",
                "factors": [
                    "Brandable suffixes: ly, fy, hub, lab, pro (+1.5)",
                    "Brandable prefixes: get, my, smart, quick (+1.0)",
                    "High character diversity (+0.5)"
                ]
            },
            "keyword_relevance": {
                "weight": "15%",
                "description": "Relevance to search terms",
                "scores": {
                    "exact_match": "5.0",
                    "contains_keyword": "3.0", 
                    "partial_matches": "0.5 each"
                }
            },
            "price_factor": {
                "weight": "5%",
                "description": "Lower price is better",
                "scores": {
                    "≤$15/year": "3.0",
                    "$16-25/year": "2.0",
                    "$26-50/year": "1.0",
                    ">$50/year": "0.5"
                }
            }
        },
        "quality_indicators": {
            "excellent": "Score 8.0-10.0 - Premium domains",
            "very_good": "Score 6.0-7.9 - Strong domains", 
            "good": "Score 4.0-5.9 - Solid domains",
            "fair": "Score 2.0-3.9 - Acceptable domains",
            "poor": "Score 0.0-1.9 - Low quality domains"
        },
        "common_words_bonus": {
            "description": "Domains containing these words get quality bonuses",
            "high_value_words": ["app", "tech", "pro", "hub", "lab", "cloud", "ai", "digital", "smart"],
            "medium_value_words": ["web", "online", "service", "data", "business", "company", "group", "team"],
            "note": "Word quality scoring helps prioritize meaningful domains over random character combinations"
        }
    }

@app.get("/api/rate-limit")
async def get_rate_limit_info(request: Request):
    """
    Get current rate limit status for your IP - **NOT rate limited**
    
    **Returns**: Your current rate limit usage and remaining requests
    """
    client_id = get_client_id(request)
    current_time = time.time()
    
    # Get current request count for this client
    with rate_limiter.lock:
        client_requests = rate_limiter.requests.get(client_id, [])
        # Clean old requests
        recent_requests = [
            req_time for req_time in client_requests
            if current_time - req_time < rate_limiter.window_seconds
        ]
        
        remaining = rate_limiter.max_requests - len(recent_requests) 
        
        # Calculate reset time
        if recent_requests:
            oldest_request = min(recent_requests)
            reset_time = oldest_request + rate_limiter.window_seconds
            reset_in_seconds = max(0, int(reset_time - current_time))
        else:
            reset_in_seconds = 0
    
    return {
        "rate_limit": {
            "max_requests": rate_limiter.max_requests,
            "window": "1 minute",
            "remaining": max(0, remaining),
            "used": len(recent_requests),
            "reset_in_seconds": reset_in_seconds,
            "client_id": client_id[:10] + "..." if len(client_id) > 10 else client_id
        },
        "note": "Rate limiting is per IP address. Use the /api/health endpoint to check status without using your quota."
    }

@app.get("/api/docs/quick-start")
async def quick_start_guide():
    """
    Quick start guide for the Domains API - **NOT rate limited**
    
    **Returns**: Step-by-step guide for getting started
    """
    return {
        "quick_start": {
            "step_1": {
                "title": "Choose Your Use Case",
                "options": {
                    "generate_suggestions": "I want AI to suggest domains for my business idea",
                    "check_specific": "I want to check if specific domains are available", 
                    "check_variations": "I have a name, show me different TLD options"
                }
            },
            "step_2": {
                "title": "Make Your First Request",
                "business_idea_example": {
                    "url": "POST /api/domains/suggest",
                    "body": {
                        "input_text": "online fitness coaching platform",
                        "input_type": "idea",
                        "field": "fitness",
                        "style": "brandable",
                        "num_choices": 5
                    }
                },
                "specific_domain_example": {
                    "url": "POST /api/domains/suggest", 
                    "body": {
                        "input_text": "mycompany.com",
                        "input_type": "exact_name",
                        "additional_domains": ["mycompany.io", "mycompany.ai"]
                    }
                }
            },
            "step_3": {
                "title": "Understand the Response",
                "response_structure": {
                    "domains": "Array of domain results with availability and pricing",
                    "score": "Quality ranking (0-10, higher is better)",
                    "ranking_factors": "Detailed breakdown of why domain got this score",
                    "registrar": "Where you can buy the domain",
                    "search_summary": "Details about your search"
                }
            },
            "step_4": {
                "title": "Rate Limits & Best Practices",
                "limits": f"{RATE_LIMIT_REQUESTS_PER_MINUTE} requests per minute per IP",
                "tips": [
                    "Use name.com provider for fastest results (default)", 
                    "Avoid porkbun unless checking individual domains",
                    "Check /api/rate-limit to see your usage",
                    "Use /api/health for status checks (doesn't count against limit)"
                ]
            }
        },
        "common_patterns": {
            "startup_domains": {
                "input_text": "your startup idea",
                "style": "brandable",
                "domain_preference": ".com",
                "num_choices": 10
            },
            "check_company_name": {
                "input_text": "yourcompanyname",
                "input_type": "base_name", 
                "domain_preference": "any",
                "num_choices": 15
            },
            "premium_search": {
                "input_text": "your business concept",
                "max_price": 100.0,
                "domain_preference": ".com",
                "provider_preference": "any"
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