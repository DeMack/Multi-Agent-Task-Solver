# Architecture Decision Records — Multi-Agent Task Solver

> Status legend: `accepted` | `proposed` | `superseded` | `deprecated`

---

## Decision Index

| ADR | Title | Status |
|---|---|---|
| [ADR-001](#adr-001-language--runtime) | Language & Runtime | accepted |
| [ADR-002](#adr-002-llm-provider--model) | LLM Provider & Model | accepted |
| [ADR-003](#adr-003-agent-orchestration-pattern) | Agent Orchestration Pattern | accepted |
| [ADR-004](#adr-004-agent-framework--build-vs-buy) | Agent Framework — Build vs Buy | accepted |
| [ADR-005](#adr-005-agent-roles-specialized-agents) | Agent Roles (Specialized Agents) | accepted |
| [ADR-006](#adr-006-context--state-sharing-between-agents) | Context & State Sharing Between Agents | accepted |
| [ADR-007](#adr-007-failure-handling--anti-hallucination) | Failure Handling & Anti-Hallucination | accepted |
| [ADR-008](#adr-008-clarification-handling-ambiguous-input) | Clarification Handling (Ambiguous Input) | accepted |
| [ADR-009](#adr-009-visibility--progress-streaming) | Visibility & Progress Streaming | accepted |
| [ADR-010](#adr-010-code-execution-sandbox) | Code Execution Sandbox | accepted |
| [ADR-011](#adr-011-web-search-provider) | Web Search Provider | accepted |
| [ADR-012](#adr-012-chart-output-format) | Chart Output Format | accepted |

---

## ADR-001: Language & Runtime

**Status:** accepted

**Context:**
Need a primary language for the prototype. Key considerations are AI/ML ecosystem maturity, available agent frameworks, and speed of prototyping within the 24h window.

**Decision:**
Python 3.13 (system default `python3`).

**Reasoning:**
- Dominant language for LLM tooling; all major agent frameworks (LangChain, LangGraph, CrewAI) are Python-first.
- Best-in-class libraries for the tools agents will need (code execution, data processing, charting).
- Fastest path to a working prototype given the constraint.
- 3.13 is the system default (`/usr/local/bin/python3`); using it avoids any path ambiguity.

**Consequences:**
- Async support via `asyncio` is mature enough for streaming/parallel agent execution.
- TypeScript would have been a viable alternative for a production system with a richer UI, but the marginal gain doesn't justify the context switch in 24h.

---

## ADR-002: LLM Provider & Model

**Status:** accepted

**Context:**
Need to choose which LLM backs the agents. Options: Anthropic Claude, OpenAI GPT-4o, open-source local models. Key criteria: tool use reliability, structured output support, context length, streaming support.

**Decision:**
Anthropic Claude (claude-sonnet-4-5 or claude-opus-4 depending on agent role).

**Reasoning:**
- Claude's tool use (function calling) is best-in-class for complex, multi-step agent tasks.
- Extended thinking on Opus makes it ideal for the Planner role where reasoning quality matters most.
- Native support for structured output (JSON mode) satisfies the aggregation requirement cleanly.
- Sonnet provides a good cost/capability tradeoff for worker agents that don't need deep reasoning.

**Trade-offs:**
- Vendor lock-in. Mitigated by wrapping all LLM calls behind a thin abstraction so the provider can be swapped.
- Cost. Opus is expensive; we use it only for planning, not execution.

---

## ADR-003: Agent Orchestration Pattern

**Status:** accepted

**Context:**
The core design question: how does the system decompose a user request and coordinate multiple agents? Options:
1. **Static pipeline** — fixed sequence of agents regardless of input.
2. **Dynamic planner → executor** — an LLM Planner agent dynamically determines which agents are needed and in what order.
3. **Reactive/event-driven** — agents subscribe to events and self-select tasks.
4. **Graph-based** (LangGraph-style) — agents are nodes in a directed graph; edges encode dependencies.

**Decision:**
Dynamic planner → executor with an explicit shared task graph.

**Reasoning:**
- A static pipeline can't satisfy the requirement that the system "decides which agents are needed" per request.
- A reactive model is complex to debug and makes progress visibility harder — bad fit for the 5-requirement visibility constraint.
- Dynamic planner + executor is the simplest pattern that satisfies all five core requirements, and it's easy to explain/demo.
- We track an explicit task graph (adjacency list of subtasks with dependencies) so parallel-safe subtasks can run concurrently and the UI can show granular progress.

**Consequences:**
- Planner is the single point of failure / quality bottleneck. Mitigated by using the strongest available model there and validating its output schema before execution begins.
- Must handle partial failures (one executor agent fails) gracefully — define a recovery strategy (see ADR-007).

---

## ADR-004: Agent Framework — Build vs Buy

**Status:** accepted

**Context:**
Many agent frameworks exist (LangChain, LangGraph, CrewAI, AutoGen, Haystack). Using one speeds up development but obscures the orchestration logic from reviewers, and most introduce significant abstraction overhead that can make the code harder to follow.

**Decision:**
Build a lightweight custom orchestrator on top of the raw Anthropic SDK. No agent framework dependency.

**Reasoning:**
- The assignment explicitly says "demonstrate orchestration logic" — a CrewAI wrapper hides that logic behind its own abstractions.
- The scope is small enough (one planner + a small set of specialized executors) that a framework buys little.
- A custom orchestrator is far easier to instrument for the visibility/streaming requirement.
- Keeps the dependency tree small and the code readable during a review.

**Trade-offs:**
- More code to write vs. using a framework. Acceptable given 24h constraint and small scope.
- We forgo battle-tested retry logic, etc. — we implement only what we need.

---

## ADR-005: Agent Roles (Specialized Agents)

**Status:** accepted

**Context:**
The system needs a fixed registry of agent types that the Planner can dispatch to. The set should be expressive enough to handle varied business requests while staying minimal.

**Decision:**
Define the following agent roles:

| Agent | Role | Tools |
|---|---|---|
| **Orchestrator** | Accepts user input, runs clarification if needed, invokes Planner, aggregates final result | None (LLM only) |
| **Planner** | Decomposes request into a typed task graph with agent assignments | None (structured output) |
| **ResearchAgent** | Retrieves information from external sources | Web search API (one tool) |
| **CodeAgent** | Writes and executes Python code (data analysis, chart generation) | Python sandbox executor (one tool) |
| **SummaryAgent** | Synthesises text from prior agent outputs into prose | None (LLM only) |
| **AggregatorAgent** | Merges all agent outputs into the final structured response | None (LLM only) |

**Constraint — one tool per agent (invariant):**
Each agent that has tool access is permitted exactly one tool. This is a hard architectural rule, not a coincidence of the current design. It enforces the single-responsibility principle at the tool level: an agent with multiple tools can be split into multiple agents, each with one. This is why ResearchAgent and CodeAgent are separate and must never be combined.

**Reasoning:**
- Covers the example use case ("Summarize financial trends + create a chart") end-to-end: Research → Code → Summary → Aggregator.
- Each role has a clear, testable responsibility — avoids the "god agent" anti-pattern.
- The set is small enough to implement fully in 24h.

**Stretch goal:**
- A dedicated ValidationAgent that checks outputs for hallucinations. Deferred — basic mitigations in ADR-007 are sufficient for the prototype.

---

## ADR-006: Context & State Sharing Between Agents

**Status:** accepted

**Context:**
Agents need access to the outputs of prior agents. Options:
1. Pass full conversation history to every agent.
2. Maintain a shared `TaskContext` object that agents read from / write to.
3. Agents message each other directly (peer-to-peer).

**Decision:**
Shared `TaskContext` object managed by the Orchestrator. Agents receive only the context slice relevant to their task.

**Reasoning:**
- Full history injection bloats each LLM call and risks confusing agents with irrelevant prior work.
- Peer-to-peer messaging requires a message broker and is overkill for this scope.
- A centrally-managed context object is simple, auditable, and aligns with the orchestrator pattern chosen in ADR-003.
- Selective context injection (only pass what the agent needs) reduces hallucination risk by limiting noise in the prompt.

**`TaskContext` structure (draft):**
```python
@dataclass
class TaskContext:
    original_request: str
    clarifications: list[str]        # from clarification phase
    plan: list[SubTask]              # from Planner
    agent_outputs: dict[str, Any]    # keyed by subtask_id
    status: dict[str, TaskStatus]    # per subtask
```

---

## ADR-007: Failure Handling & Anti-Hallucination

**Status:** accepted

**Context:**
LLM agents can hallucinate, loop, or fail. The assignment explicitly calls out hallucination and repetition as pitfalls to address.

**Decision:**
- **Schema validation:** Planner output is validated against a Pydantic schema before execution begins. If invalid, Planner is re-prompted once with the error; if it fails again, the request is rejected with a clear user-facing message.
- **Tool result grounding:** CodeAgent and ResearchAgent results are actual tool outputs (code execution results, search snippets), not LLM-generated claims — this is the primary hallucination mitigation.
- **Repetition detection:** Before each agent call, check if an identical subtask has already produced output in `TaskContext`. Skip if so.
- **Max retries:** Each agent gets at most 2 retries on failure before the subtask is marked `failed` and aggregation proceeds with a partial result.
- **Timeout:** Each agent invocation has a wall-clock timeout (configurable, default 60s).

---

## ADR-008: Clarification Handling (Ambiguous Input)

**Status:** accepted

**Context:**
The "high marks" criterion is that agents can ask clarifying questions before proceeding when the request is ambiguous or incomplete.

**Decision:**
The Orchestrator runs a pre-planning clarification phase:
1. Pass the user request to the Orchestrator LLM with a prompt instructing it to identify any ambiguities that would materially affect the plan.
2. If ambiguities are found, surface them to the user as numbered questions before planning begins.
3. User answers are appended to `TaskContext.clarifications` and included in the Planner's context.
4. If the request is clear enough, skip clarification and proceed directly to planning.

**Reasoning:**
- Separating clarification from planning keeps each step focused and avoids mid-execution interruptions.
- Structured numbered questions are easier for users to answer than open-ended dialogue.
- The skip condition ensures non-ambiguous requests are fast.

**Out of scope:**
- Mid-execution clarification (stretch goal, deferred).

---

## ADR-009: Visibility & Progress Streaming

**Status:** accepted

**Context:**
Requirement 5 says users must see progress/status updates for each agent. Options:
- CLI with live output (simplest).
- Web UI with Server-Sent Events (SSE) streaming.
- WebSocket-based chat-style interface.

**Decision:**
FastAPI backend with SSE streaming + a minimal HTML/JS frontend. Backend and frontend must be strictly decoupled: the frontend communicates with the backend only via documented HTTP/SSE endpoints. No server-side templating or tight coupling. This allows the frontend to be replaced independently.

**Reasoning:**
- SSE is simpler than WebSockets for unidirectional server-to-client progress events.
- A web UI is dramatically more impressive in the 5-minute demo than a CLI.
- FastAPI is async-native, making streaming straightforward.
- The frontend can be a single HTML file (no build step) to keep scope tight while still being swappable.
- Strict decoupling future-proofs the interface: a React/Vue frontend, CLI client, or VS Code extension could consume the same API without touching backend code.

**Event types emitted (SSE):**
- `clarification_needed` — Orchestrator has questions for the user.
- `plan_ready` — Planner has produced a task graph.
- `agent_started` — A subtask agent has begun.
- `agent_completed` — A subtask agent finished (includes output summary).
- `agent_failed` — A subtask agent failed.
- `result_ready` — Final aggregated result is available.

**Non-standard dependencies introduced by this decision:**

| Package | Why required |
|---|---|
| `fastapi` | The web framework itself — provides routing, request/response parsing, and OpenAPI schema generation. |
| `uvicorn[standard]` | ASGI server that runs the FastAPI app and listens for HTTP connections. FastAPI has no built-in server. |
| `sse-starlette` | FastAPI has no native SSE primitive. This package adds an `EventSourceResponse` that handles SSE framing and keep-alive over a standard async generator. |
| `starlette` | Pulled in automatically as a FastAPI dependency; not used directly. |
| `httpx` *(dev only)* | Required by Starlette's `TestClient` to make requests against the app in tests. Not used at runtime. |

---

## ADR-010: Code Execution Sandbox

**Status:** accepted

**Context:**
CodeAgent needs to execute Python code. This is a security-sensitive capability. Options:
- `exec()` in-process (dangerous).
- Subprocess with timeout.
- Docker container (secure, complex to set up in 24h).
- External service (e.g., Pyodide, Judge0 API).

**Decision:**
Subprocess execution with a strict timeout and restricted working directory, for the prototype.

**Reasoning:**
- Docker is the right production answer but adds significant setup overhead for a local prototype.
- An external API (Judge0) adds a network dependency and signup friction for reviewers running the demo.
- Subprocess + timeout is sufficient to demonstrate the capability safely in a local/demo context.
- We explicitly document this as a production trade-off in the README.

**Mitigations:**
- Execution timeout (default 30s).
- Separate working directory per run, cleaned up after execution.
- No network access from the subprocess (block via environment, not enforced in prototype).

**Stretch goal:**
- Surface a warning to the user when the timeout is approaching and allow them to extend it. Some tasks (e.g., large data processing) genuinely need more time. Deferred until core flow is working.

---

## ADR-011: Web Search Provider

**Status:** accepted

**Context:**
ResearchAgent needs a search tool. Options: Brave Search API, Tavily, SerpAPI, DuckDuckGo (unofficial/free).

**Decision:**
DuckDuckGo via the `duckduckgo-search` Python library (no API key required).

**Reasoning:**
- Zero signup/API key friction for reviewers running the demo locally — important for a take-home prototype.
- The `duckduckgo-search` library is a well-maintained unofficial client.
- Sufficient result quality for the demo use cases.

**Trade-offs:**
- Rate limits are unofficial and unpredictable. Acceptable for a demo.
- Not suitable for production. Documented as a known trade-off in the README.

**Stretch goal:**
- Make the search provider configurable via an environment variable or config file, allowing Tavily/Brave/SerpAPI to be plugged in without code changes.

---

## ADR-012: Chart Output Format

**Status:** accepted

**Context:**
CodeAgent may produce charts as part of its output. The frontend needs to display them. Options: PNG file served as a static asset, base64-encoded inline, SVG string.

**Decision:**
PNG file written to a per-run output directory, served as a static file by FastAPI, and referenced by URL in the agent output.

**Reasoning:**
- `matplotlib` (Python's dominant charting library) produces PNG by default — zero extra work.
- Serving a static file is trivial in FastAPI (`StaticFiles` mount).
- A URL reference in the agent output keeps `TaskContext` lightweight (no large base64 blobs in memory).
- The frontend renders it with a standard `<img>` tag.

---

## Stretch Goals Backlog

| # | Goal | Depends on |
|---|---|---|
| S1 | Mid-execution live conversation with orchestrator | ADR-008 accepted core first |
| S2 | Multi-turn refinement (user modifies request after output) | Core flow complete |
| S3 | ValidationAgent for hallucination checking | Core agents complete |
| S4 | Configurable search provider (env var / config) | ADR-011 |
| S5 | Timeout extension — warn user before expiry, allow extension | ADR-010 |
