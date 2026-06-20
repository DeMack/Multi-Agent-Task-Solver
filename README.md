# Multi-Agent Task Solver

A web application that accepts a plain-language business request, breaks it into subtasks using Claude, dispatches those subtasks to specialized AI agents, and streams live progress back to the browser via Server-Sent Events.

## Prerequisites

- Python 3.13+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install runtime and dev dependencies
pip install -e ".[dev]"

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the server

```bash
# Activate the virtual environment first (uvicorn is installed inside it)
source .venv/bin/activate

uvicorn src.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Configuration

All settings are read from environment variables at startup.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `AGENT_TIMEOUT_SECONDS` | `120` | Per-agent wall-clock timeout |
| `CODE_EXECUTION_TIMEOUT_SECONDS` | `30` | Subprocess sandbox timeout |
| `MAX_AGENT_RETRIES` | `2` | Retry attempts before marking a subtask failed |
| `OUTPUTS_DIR` | `outputs` | Directory for generated files (charts, etc.) |

## Running tests

```bash
# Unit tests only (no API calls, fast) — coverage report printed to terminal
pytest

# Open HTML coverage report in browser
open htmlcov/index.html

# Include integration tests (calls real Anthropic API, takes several minutes)
pytest --integration

# Run just the end-to-end test
pytest --integration tests/integration/test_e2e.py -v -s

# Lint and type-check
ruff check .
ruff format --check .
pyright src/
```

## Architecture

```
Browser (SSE client)
       │
       │  POST /task                  submit request
       │  GET  /task/{id}/stream      live events (SSE)
       │  POST /task/{id}/clarify     answer clarification questions
       │  POST /task/{id}/message     send a mid-run steering message (S1)
       │  POST /task/{id}/refine      refine result after completion (S2)
       │  GET  /outputs/{file}        generated files (charts)
       ▼
FastAPI (src/main.py)
       │
       ▼
Orchestrator (src/orchestrator.py)
   1. Clarification phase — asks Claude if the request is ambiguous;
      emits clarification_needed and waits for user answers if so
   2. Planning phase — Planner (Claude Opus) produces a task graph:
      a list of subtasks with agent assignments and dependency edges
   3. Execution phase — walks the graph in dependency order;
      runs independent subtasks concurrently with asyncio.gather;
      each subtask has a per-agent timeout and up to N retries
   4. Aggregation — AggregatorAgent merges all outputs into a
      structured final result {answer, artifacts, warnings}

Agents:
  - Planner       Claude Opus — produces the task graph
  - ResearchAgent Claude Sonnet + DuckDuckGo search
  - CodeAgent     Claude Sonnet + subprocess sandbox (writes and runs Python)
  - SummaryAgent  Claude Sonnet — synthesizes prose from prior outputs
  - AggregatorAgent Claude Sonnet — merges everything into the final result

SSE events (in order):
  clarification_needed → plan_ready → agent_started / agent_completed /
  agent_failed / agent_skipped / agent_restarted (×N) →
  [user_message_ack / plan_reset / plan_ready (on steer)] →
  result_ready
```

## 24h trade-offs

These are things deliberately left out or simplified due to the time constraint. Each is a known gap, not an oversight.

**No persistent storage.** Task state lives in memory and the local filesystem. The server loses all in-flight tasks on restart. A production system would use Redis for the event queue and a database for task history and output metadata.

**DuckDuckGo search (snippet-only).** The research agent uses the free DuckDuckGo API, which returns short title/URL/snippet excerpts — not full page content. Research quality is therefore limited by what fits in a snippet. A production deployment would use a paid API (Exa, Tavily, Bing) that returns full document text.

**No authentication or multi-tenancy.** Any client can submit tasks and read any task's SSE stream by guessing the UUID. Adding auth was out of scope.

**No horizontal scaling.** The event queue is an in-process `asyncio.Queue`. A second server instance would have no visibility into another instance's tasks. Redis pub/sub or a message broker would be needed for scale-out.

**Frontend is a single HTML file.** No build step, no bundler, no component framework. This was the fastest path to a working UI; it would need to be replaced for a production-grade frontend.

**Summary agent is largely passive.** The `summary` agent type exists in the planner's vocabulary but the orchestrator includes a planning rule that prevents it being used after a single research step (research agents already synthesize). It's only activated when two or more prior subtasks need narrative merging.

---

## Design decisions

**Single event queue per task, not WebSocket.**
SSE is unidirectional and simpler to implement correctly than WebSocket. The only bidirectional interaction (clarification answers) goes through a separate REST endpoint, keeping the streaming path read-only and stateless from the client's perspective.

**asyncio.to_thread for SDK calls.**
The Anthropic Python SDK is synchronous. Wrapping calls in `asyncio.to_thread` lets the async event loop remain responsive while agents run, enabling true concurrent subtask execution via `asyncio.gather` without a thread-per-request model.

**Module-level queue registry with _reset() for tests.**
Using a module-level dict avoids passing a registry object through every layer while still being fully testable — `_reset()` gives each test a clean slate without spawning new processes or reloading modules.

**CodeAgent uses a subprocess sandbox.**
Python code generated by Claude is executed in a separate subprocess with a configurable timeout and an isolated working directory. The subprocess cannot import from the host environment's src package. Stdout/stderr are captured and returned to Claude for inspection before producing a result.

**Planner retries once on schema validation failure.**
The task graph must conform to a Pydantic model (valid agent names, no self-referential dependencies). If Claude returns invalid JSON or an invalid graph, the planner makes one more attempt with the validation error attached to the prompt.

