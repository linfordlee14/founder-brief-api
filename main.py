import os
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from agent import FounderBrief, run_research

@asynccontextmanager
async def lifespan(app: FastAPI):
    logfire.instrument_fastapi(app)
    yield

app = FastAPI(
    title="Founder Brief API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Lovable apps and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.lovable\.app|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BriefRequest(BaseModel):
    topic: str

@app.post("/api/brief", response_model=FounderBrief)
async def generate_brief(request: BriefRequest) -> FounderBrief:
    """Generate a structured founder brief for a given topic."""
    if not request.topic or len(request.topic.strip()) < 3:
        raise HTTPException(status_code=422, detail="Topic must be at least 3 characters.")
    try:
        brief = await run_research(request.topic.strip())
        return brief
    except Exception as exc:
        logfire.error("Brief generation failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "founder-brief-api"}
