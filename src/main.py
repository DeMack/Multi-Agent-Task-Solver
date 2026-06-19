import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import config
from src.models import (
    ClarifyRequest,
    ClarifyResponse,
    CreateTaskRequest,
    CreateTaskResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Multi-Agent Task Solver", lifespan=lifespan)

app.mount("/outputs", StaticFiles(directory=config.outputs_dir, check_dir=False), name="outputs")


@app.post("/task", response_model=CreateTaskResponse)
async def create_task(body: CreateTaskRequest) -> CreateTaskResponse:
    task_id = str(uuid.uuid4())
    # Orchestrator invocation will be wired in Phase 3.
    return CreateTaskResponse(task_id=task_id, status="pending")


@app.post("/task/{task_id}/clarify", response_model=ClarifyResponse)
async def submit_clarification(task_id: str, body: ClarifyRequest) -> ClarifyResponse:
    # Orchestrator resume will be wired in Phase 3.
    return ClarifyResponse(task_id=task_id, status="resumed")


@app.get("/task/{task_id}/stream")
async def stream_task(task_id: str) -> StreamingResponse:
    # SSE event queue will be wired in Phase 3.
    async def empty_stream():
        yield "data: {}\n\n"

    return StreamingResponse(empty_stream(), media_type="text/event-stream")
