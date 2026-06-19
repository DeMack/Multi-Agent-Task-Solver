import asyncio
import uuid
from contextlib import asynccontextmanager

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.events import create_queue, get_queue, submit_clarification
from src.models import (
    ClarifyRequest,
    ClarifyResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskContext,
)
from src.orchestrator import Orchestrator

_client = Anthropic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Multi-Agent Task Solver", lifespan=lifespan)

app.mount("/outputs", StaticFiles(directory=config.outputs_dir, check_dir=False), name="outputs")


@app.post("/task", response_model=CreateTaskResponse)
async def create_task(body: CreateTaskRequest) -> CreateTaskResponse:
    task_id = str(uuid.uuid4())
    context = TaskContext(
        task_id=task_id,
        original_request=body.request,
        clarifications=body.clarifications,
    )
    create_queue(task_id)
    asyncio.create_task(Orchestrator(_client, config).run(context))
    return CreateTaskResponse(task_id=task_id, status="pending")


@app.post("/task/{task_id}/clarify", response_model=ClarifyResponse)
async def clarify_task(task_id: str, body: ClarifyRequest) -> ClarifyResponse:
    submit_clarification(task_id, body.answers)
    return ClarifyResponse(task_id=task_id, status="resumed")


@app.get("/task/{task_id}/stream")
async def stream_task(task_id: str) -> EventSourceResponse:
    queue = get_queue(task_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield {"data": item.model_dump_json(), "event": item.event}

    return EventSourceResponse(event_generator())
