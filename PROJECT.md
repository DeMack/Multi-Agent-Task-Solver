# Multi-Agent Task Solver — Project Design & Plan

> Decisions and their rationale live in `ADR.md`. This file is the working design and task tracker.

---

## Contents

- [Multi-Agent Task Solver — Project Design \& Plan](#multi-agent-task-solver--project-design--plan)
  - [Contents](#contents)
  - [What We're Building](#what-were-building)
  - [System Architecture](#system-architecture)
  - [Components](#components)
    - [Backend (`src/`)](#backend-src)
    - [Frontend (`static/`)](#frontend-static)
    - [Output directory (`outputs/`)](#output-directory-outputs)
  - [Data Flow](#data-flow)
  - [Key Data Models](#key-data-models)
  - [SSE Event Schema](#sse-event-schema)
  - [Implementation Plan](#implementation-plan)
    - [Phase 0 — Scaffold](#phase-0--scaffold)
    - [Phase 1 — Tools](#phase-1--tools)
    - [Phase 2 — Agents](#phase-2--agents)
    - [Phase 3 — Orchestrator](#phase-3--orchestrator)
    - [Phase 4 — API \& Streaming](#phase-4--api--streaming)
    - [Phase 5 — Frontend](#phase-5--frontend)
    - [Phase 6 — Polish \& Docs](#phase-6--polish--docs)
  - [Stretch Goals (not in plan above)](#stretch-goals-not-in-plan-above)

---

## What We're Building

A web application that accepts a plain-language business request, breaks it into subtasks via a Planner agent, dispatches those subtasks to specialized executor agents, and streams live progress back to the user — returning a final structured result when all agents complete.

---

## System Architecture

```
Browser (HTML/JS)
      │
      │  POST /task                (submit request + clarification answers)
      │  GET  /task/{id}/stream        (SSE — live progress events)
      │  POST /task/{id}/clarify       (answer clarification questions)
      │  POST /task/{id}/message       (mid-run steering — S1)
      │  POST /task/{id}/refine        (post-result refinement — S2)
      │  GET  /outputs/{file}          (static files — charts, etc.)
      ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              Orchestrator                    │   │
│  │                                              │   │
│  │  1. Clarification phase (pre-planning)       │   │
│  │  2. Invoke Planner → get task graph          │   │
│  │  3. Execute subtasks (respecting deps)       │   │
│  │  4. Emit SSE events at each state change     │   │
│  │  5. Invoke Aggregator → final result         │   │
│  └──────────┬───────────────────────────────────┘   │
│             │  TaskContext (shared state)           │
│    ┌────────┼────────────────────────────┐          │
│    │        │                            │          │
│    ▼        ▼                            ▼          │
│  Planner  ResearchAgent  CodeAgent  SummaryAgent    │
│             │                │                      │
│         DuckDuckGo       subprocess                 │
│         search lib       (Python sandbox)           │
│                                                     │
│  AggregatorAgent  (final merge)                     │
└─────────────────────────────────────────────────────┘
```

---

## Components

### Backend (`src/`)

| Module | Responsibility |
|---|---|
| `main.py` | FastAPI app, route definitions |
| `orchestrator.py` | Top-level coordination: clarification → plan → execute → aggregate |
| `agents/planner.py` | Calls Claude (Opus) to produce a validated task graph from the request |
| `agents/research.py` | Calls Claude (Sonnet) with DuckDuckGo search tool |
| `agents/code.py` | Calls Claude (Sonnet) to write + execute Python in a subprocess sandbox |
| `agents/summary.py` | Calls Claude (Sonnet) to synthesize prose from prior outputs |
| `agents/aggregator.py` | Calls Claude (Sonnet) to merge all outputs into a final structured response |
| `tools/search.py` | DuckDuckGo search wrapper (one tool, used only by ResearchAgent) |
| `tools/executor.py` | Subprocess sandbox runner (one tool, used only by CodeAgent) |
| `models.py` | Pydantic models: `SubTask`, `TaskGraph`, `TaskContext`, `TaskStatus`, SSE event schemas |
| `events.py` | SSE event emitter — thin wrapper that formats and queues events per task |
| `config.py` | Environment/config loading (`ANTHROPIC_API_KEY`, timeouts, output dir, etc.) |

### Frontend (`static/`)

| File | Responsibility |
|---|---|
| `index.html` | Single-page UI — request input, clarification Q&A, live agent progress feed, final result display |

### Output directory (`outputs/`)

Per-run directories for generated files (charts, etc.), served as static assets by FastAPI.

---

## Data Flow

```
User submits request
        │
        ▼
Orchestrator: clarification check
        │
        ├─ ambiguous → emit `clarification_needed` → wait for user answers → resume
        │
        └─ clear → continue
        │
        ▼
Planner: produce TaskGraph (list of SubTasks with deps + agent assignments)
        │
        └─ validated against Pydantic schema (retry once on failure)
        │
        ▼
Orchestrator: walk task graph
        │
        ├─ emit `plan_ready`
        │
        └─ for each SubTask (respecting dependency ordering):
              │
              ├─ emit `agent_started`
              ├─ inject relevant TaskContext slice into agent
              ├─ run agent (with timeout + 2 retries)
              ├─ emit `agent_completed` or `agent_failed`
              └─ write output to TaskContext.agent_outputs[subtask_id]
        │
        ▼
AggregatorAgent: merge all outputs → structured final result
        │
        └─ emit `result_ready`
```

---

## Key Data Models

```python
class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

class SubTask(BaseModel):
    id: str
    description: str
    agent: Literal["research", "code", "summary", "aggregator"]
    depends_on: list[str]   # IDs of subtasks that must complete first

class TaskGraph(BaseModel):
    subtasks: list[SubTask]

class TaskContext(BaseModel):
    task_id: str
    original_request: str
    clarifications: list[str]
    plan: TaskGraph | None = None
    agent_outputs: dict[str, Any] = {}
    user_messages: list[str] = []      # accumulated mid-run steering messages (S1) and refinement messages (S2)
    prior_results: list[dict] = []     # aggregator outputs from prior runs (S2)
```

---

## SSE Event Schema

All events share a common envelope:

```json
{
  "event": "<event_type>",
  "task_id": "<uuid>",
  "timestamp": "<iso8601>",
  "data": { ... }
}
```

| `event` | `data` fields |
|---|---|
| `clarification_needed` | `questions: list[str]` |
| `plan_ready` | `subtasks: list[{id, description, agent, depends_on}]` |
| `agent_started` | `subtask_id`, `agent`, `description` |
| `agent_completed` | `subtask_id`, `agent`, `summary` (brief text), `has_artifact: bool` |
| `agent_failed` | `subtask_id`, `agent`, `error` |
| `agent_skipped` | `subtask_id` |
| `agent_restarted` | `subtask_id` |
| `user_message_ack` | `acknowledgment: str`, `reuse_plan: bool`, `restarted_subtask_ids: list[str]`, `skipped_subtask_ids: list[str]` |
| `plan_reset` | *(empty — signals UI to clear plan before second `plan_ready`)* |
| `result_ready` | `result: {answer: str, artifacts: list[{type, url}], warnings: list[str]}` |

---

## Implementation Plan

### Phase 0 — Scaffold
- [x] Create project directory structure
- [x] Set up `pyproject.toml` / `requirements.txt` with dependencies
- [x] `config.py` — env loading, constants
- [x] `models.py` — all Pydantic models
- [x] `main.py` — FastAPI app skeleton with route stubs

### Phase 1 — Tools
- [x] `tools/search.py` — DuckDuckGo wrapper, unit-testable in isolation
- [x] `tools/executor.py` — subprocess sandbox: run code, capture stdout/stderr, enforce timeout, clean up working dir

### Phase 2 — Agents
- [x] `agents/planner.py` — Claude Opus call, structured output, Pydantic validation + one retry
- [x] `agents/research.py` — Claude Sonnet + search tool
- [x] `agents/code.py` — Claude Sonnet + executor tool (writes code, executes, returns result + artifact path)
- [x] `agents/summary.py` — Claude Sonnet, text synthesis
- [x] `agents/aggregator.py` — Claude Sonnet, merge outputs into final structured result

### Phase 3 — Orchestrator
- [x] `events.py` — SSE event queue per task
- [x] `orchestrator.py` — clarification phase, task graph execution loop, dependency ordering, timeout + retry logic

### Phase 4 — API & Streaming
- [x] `POST /task` — accept request, create TaskContext, kick off orchestrator in background task, return `task_id`
- [x] `POST /task/{id}/clarify` — submit clarification answers, resume orchestrator
- [x] `GET /task/{id}/stream` — SSE endpoint, stream events from queue
- [x] `GET /outputs/{path}` — serve static output files

### Phase 5 — Frontend
- [x] `static/index.html` — request form, SSE listener, agent progress cards, clarification Q&A form, result display with artifact rendering

### Phase 6 — Polish & Docs
- [x] End-to-end test with the example request ("Summarize financial trends + create a chart")
- [x] `README.md` — setup, how to run, design decisions, trade-offs

---

## Stretch Goals (not in plan above)

| ID | Goal | Status | ADR |
|---|---|---|---|
| S1 | Mid-execution live conversation with orchestrator | ✅ Done | ADR-014 |
| S2 | Multi-turn refinement (user modifies request after output) | ✅ Done | ADR-015 |
| S3 | ValidationAgent for hallucination checking | — | — |
| S4 | Configurable search provider via env var | — | — |
| S5 | Timeout extension — warn + allow user to add time | — | — |
| S6 | Late-result reuse — when a retry starts after a timeout, monitor the original background thread; if it completes before the retry does, use its output and cancel the retry | — | — |
| S7 | FetchAgent — full-page content retrieval to complement ResearchAgent snippets | — | — |
