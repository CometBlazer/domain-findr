# domainFindr

AI-powered domain name suggestion API built with FastAPI and Porkbun integration.

## Setup

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys
5. `uvicorn src.main:app --reload`

## API Endpoint

- `POST /api/domains/suggest` - Get domain suggestions
