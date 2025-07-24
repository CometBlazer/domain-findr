# Snapsi

Domain name suggestion API built with FastAPI and Porkbun integration.

## Setup

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys
5. `uvicorn src.main:app --reload --port 8000`
6. `curl http://localhost:8000/api/health`

## Quick Commands

```bash
# Development
uvicorn src.main:app --reload

# Production test
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Docker build
docker build -t domain-findr .

# Docker run
docker run -p 8000:8000 domain-findr
```

## API Endpoints

### Main Endpoints

- `POST /api/domains/suggest` - Get domain suggestions with intelligent ranking
- `POST /api/parse-input` - Debug input parsing and auto-detection

### Information Endpoints (Not Rate Limited)

- `GET /api/health` - API status and configuration
- `GET /api/pricing` - Pricing and provider information
- `GET /api/examples` - Example API requests for all use cases
- `GET /api/tlds` - List of all supported Top Level Domains
- `GET /api/ranking` - Detailed domain ranking algorithm information
- `GET /api/rate-limit` - Current rate limit status for your IP
- `GET /api/docs/quick-start` - Step-by-step quick start guide

### Testing Endpoints (Rate Limited)

- `GET /api/test-providers` - Test all configured domain providers
- `GET /api/test-porkbun` - Test Porkbun API connection
- `GET /api/test-namecom` - Test Name.com API connection

## Rate Limiting

- **Limit**: 100 requests per minute per IP address
- **Rate limited endpoints**: All POST endpoints and test endpoints
- **Free endpoints**: Health, pricing, examples, TLDs, ranking info

## Features

- 🧠 **AI-Powered Suggestions** - Generate brandable domain names
- 📊 **Intelligent Ranking** - GoDaddy-style scoring (0-10 scale)
- ⚡ **Real-time Availability** - Live domain checking
- 🏢 **Multi-Provider** - Porkbun and Name.com integration
- 🎯 **Input Auto-Detection** - Automatically detects idea/base_name/exact_name
- 💰 **Price Filtering** - Set maximum price limits
- 🚀 **Fast Performance** - Bulk checking with caching