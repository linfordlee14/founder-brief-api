# Founder Brief API

FastAPI + Pydantic AI agent that researches a topic and returns a structured founder brief.

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in all API keys
3. `uvicorn main:app --host 0.0.0.0 --port 8000`

## Test
curl -X POST http://localhost:8000/api/brief \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI-powered legal contract review for SMBs"}'

## Deploy
Push to GitHub, connect to Render, set env vars in Render dashboard.
