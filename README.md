# domainFindr

AI-powered domain name suggestion API built with FastAPI and Porkbun integration.

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

## API Endpoint

- `POST /api/domains/suggest` - Get domain suggestions
