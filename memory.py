import os
from upstash_redis.asyncio import Redis

redis = Redis(
    url=os.environ["UPSTASH_REDIS_REST_URL"],
    token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
)

async def save_memory(session_id: str, label: str, value: str) -> None:
    await redis.hset(session_id, label, value)

async def get_all_memory(session_id: str) -> dict[str, str]:
    data = await redis.hgetall(session_id)
    return data or {}

async def clear_session(session_id: str) -> None:
    await redis.delete(session_id)
