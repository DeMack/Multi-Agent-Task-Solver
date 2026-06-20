import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager

from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.events import cleanup, create_queue, get_queue, submit_clarification, submit_user_message
from src.models import (
    ClarifyRequest,
    ClarifyResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskContext,
    UserMessageRequest,
    UserMessageResponse,
)
from src.orchestrator import Orchestrator

_client = Anthropic()

# Attach a dedicated handler to the src logger so INFO messages always appear
# regardless of uvicorn's root-logger level. The guard prevents duplicate
# handlers when uvicorn hot-reloads the module.
_src_log = logging.getLogger("src")
if not _src_log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s", "%H:%M:%S")
    )
    _src_log.addHandler(_h)
    _src_log.propagate = False
_src_log.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Multi-Agent Task Solver", lifespan=lifespan)

app.mount("/outputs", StaticFiles(directory=config.outputs_dir, check_dir=False), name="outputs")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse("static/index.html")


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


@app.post("/task/{task_id}/message", response_model=UserMessageResponse)
async def message_task(task_id: str, body: UserMessageRequest) -> UserMessageResponse:
    if get_queue(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if not submit_user_message(task_id, body.message):
        raise HTTPException(status_code=409, detail="Task is not accepting messages yet")
    return UserMessageResponse(task_id=task_id, status="received")


@app.get("/task/{task_id}/stream")
async def stream_task(task_id: str) -> EventSourceResponse:
    queue = get_queue(task_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": item.model_dump_json(), "event": item.event}
        finally:
            cleanup(task_id)

    return EventSourceResponse(event_generator())
