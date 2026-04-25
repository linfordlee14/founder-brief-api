import os
import uuid
from dataclasses import dataclass
from typing import Optional

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import AsyncTavilyClient

from memory import save_memory, get_all_memory

# --- Logfire MUST be configured before any agent is instantiated ---
logfire.configure(
    send_to_logfire=True if os.getenv("LOGFIRE_TOKEN") else False,
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="founder-brief-api",
)
logfire.instrument_pydantic_ai()

# --- Context passed via RunContext deps ---
@dataclass
class AgentContext:
    session_id: str
    tavily_api_key: str

# --- Structured output model ---
class FounderBrief(BaseModel):
    topic: str = Field(description="The researched topic or market")
    problem: str = Field(description="The core problem this market has, in 2-3 sentences")
    icp: str = Field(description="Ideal Customer Profile — who suffers most from this problem")
    competitors: list[str] = Field(description="Top 3-5 existing solutions or competitors")
    gtm_angle: str = Field(description="A specific go-to-market angle or beachhead market")
    why_now: str = Field(description="Why this opportunity exists right now, not 3 years ago")
    confidence_score: float = Field(description="Confidence in research quality, 0.0 to 1.0", ge=0.0, le=1.0)

# --- Tools ---
async def search_web(ctx: RunContext[AgentContext], query: str) -> str:
    """Search the web using Tavily and store the result in session memory."""
    client = AsyncTavilyClient(api_key=ctx.deps.tavily_api_key)
    response = await client.search(query, max_results=3)
    results = response.get("results", [])
    if not results:
        return f"No results found for: {query}"
    formatted = []
    for i, r in enumerate(results[:3], 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "")[:400]
        formatted.append(f"{i}. {title}\n   URL: {url}\n   {content}")
    result_text = "\n\n".join(formatted)
    # Save to memory before returning
    label = f"search:{query[:50].replace(' ', '_')}"
    await save_memory(ctx.deps.session_id, label, result_text)
    return result_text

async def recall_research(ctx: RunContext[AgentContext]) -> str:
    """Retrieve all previous research steps from session memory."""
    try:
        memories = await get_all_memory(ctx.deps.session_id)
        if not memories:
            return "No previous research steps found."
        return "\n\n---\n\n".join(f"[{k}]\n{v}" for k, v in memories.items())
    except Exception as e:
        return f"Memory unavailable, proceed with search results only. Error: {str(e)}"

# --- Agent definition ---
agent = Agent(
    "groq:llama-3.3-70b-versatile",
    deps_type=AgentContext,
    output_type=FounderBrief,
    tools=[search_web, recall_research],
    system_prompt=(
        "You are a contrarian startup analyst who helps founders avoid bad markets. "
        "Given a topic, you run 3 web searches: (1) the core problem and pain intensity, "
        "(2) existing solutions and their weaknesses, (3) recent funding or market signals. "
        "Use recall_research to synthesise all findings before writing the brief. "
        "Rules for output: "
        "- problem: be specific about WHO loses money/time and HOW MUCH. No vague statements. "
        "- icp: name a specific job title, company size, and geography. Not just 'SMBs'. "
        "- competitors: name real funded companies with their main weakness in brackets. "
        "- gtm_angle: name a specific beachhead — one city, one vertical, one use case. "
        "- why_now: cite a specific regulatory change, technology shift, or market event from the last 2 years. "
        "- confidence_score: be honest. Score below 0.7 if search results were thin. "
        "Do not use consulting speak. Write like a YC partner giving direct feedback."
    ),
)

async def run_research(topic: str) -> FounderBrief:
    """Run the agent for a given topic and return a FounderBrief."""
    session_id = f"session:{uuid.uuid4().hex[:8]}"
    deps = AgentContext(
        session_id=session_id,
        tavily_api_key=os.environ["TAVILY_API_KEY"],
    )
    result = await agent.run(
        f"Research this market and generate a founder brief: {topic}",
        deps=deps,
    )
    return result.output
