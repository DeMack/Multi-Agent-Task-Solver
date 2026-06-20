from typing import Any, Literal

from pydantic import BaseModel


class SubTask(BaseModel):
    id: str
    description: str
    agent: Literal["research", "code", "summary", "aggregator"]
    depends_on: list[str]


class TaskGraph(BaseModel):
    subtasks: list[SubTask]


class TaskContext(BaseModel):
    task_id: str
    original_request: str
    clarifications: list[str]
    plan: TaskGraph | None = None
    agent_outputs: dict[str, Any] = {}
    user_messages: list[str] = []
    prior_results: list[dict] = []


# API request / response models


class CreateTaskRequest(BaseModel):
    request: str
    clarifications: list[str] = []


class CreateTaskResponse(BaseModel):
    task_id: str
    status: str


class ClarifyRequest(BaseModel):
    answers: list[str]


class ClarifyResponse(BaseModel):
    task_id: str
    status: str


class UserMessageRequest(BaseModel):
    message: str


class UserMessageResponse(BaseModel):
    task_id: str
    status: str


class RefineRequest(BaseModel):
    message: str


class RefineResponse(BaseModel):
    task_id: str
    status: str


# SSE event envelope


class SSEEvent(BaseModel):
    event: str
    task_id: str
    timestamp: str
    data: dict[str, Any]
