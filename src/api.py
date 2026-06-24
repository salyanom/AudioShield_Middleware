"""HTTP gateway for integrating AudioShield with real voice-AI services."""

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from middleware import process_transcript


app = FastAPI(
    title="AudioShield Gateway",
    version="0.2.0",
    description="Pre-generation blocking and post-generation output mitigation.",
)


class SecureRequest(BaseModel):
    transcript: str = Field(min_length=1, description="Text produced by the STT service")
    supplied_response: str | None = Field(
        default=None,
        description="Existing model output to inspect; omit to use the configured LLM",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@app.post("/v1/secure")
async def secure(request: SecureRequest):
    try:
        result = await run_in_threadpool(
            process_transcript,
            request.transcript,
            supplied_response=request.supplied_response,
        )
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
