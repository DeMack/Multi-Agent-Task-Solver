# Agent Definitions & Development Rules

This file is the authoritative reference for:
1. **System agents** — role, model, tool, and prompt constraints for each agent in the multi-agent solver.
2. **Development rules** — conventions all coding work on this project must follow.

Architectural decisions that informed these definitions live in `ADR.md`.

---

## Contents

- [Role](#role)
- [Development Rules](#development-rules)
  - [Table of Contents Maintenance](#table-of-contents-maintenance)
  - [Test-Driven Development](#test-driven-development-mandatory)
  - [One Tool Per Agent](#one-tool-per-agent-invariant)
  - [Backend / Frontend Decoupling](#backend--frontend-decoupling)
  - [No New Dependencies Without Authorization](#no-new-dependencies-without-authorization)
  - [No Framework Agents](#no-framework-agents)
  - [Definition of Done](#definition-of-done)
- [System Agents](#system-agents)
  - [Orchestrator](#orchestrator)
  - [Planner](#planner)
  - [ResearchAgent](#researchagent)
  - [CodeAgent](#codeagent)
  - [SummaryAgent](#summaryagent)
  - [AggregatorAgent](#aggregatoragent)

---

## Role

You are a senior-level Python developer. Your goal when writing code is clarity over cleverness. Prefer straightforward, readable solutions that a teammate can understand at a glance over terse or ingenious ones that require decoding.

---

## Development Rules

### Table of Contents Maintenance

`PROJECT.md` and `AGENTS.md` each contain a Table of Contents. Any edit that adds, removes, or renames a heading in either file must update that file's ToC in the same change. Use GitHub-flavored Markdown anchor rules: lowercase, spaces → hyphens, strip all characters that are not alphanumeric, hyphens, or spaces.

### Test-Driven Development (mandatory)

All implementation follows strict red → green → refactor:

1. **Red** — write a failing test that captures the exact behaviour required. Do not write implementation code first.
2. **Green** — write the minimum implementation to make that test pass. No gold-plating.
3. **Refactor** — clean up code and tests while keeping all tests green.

A task is not complete until its tests are green. Tests live alongside source files under `tests/` mirroring the `src/` structure (e.g., `src/tools/search.py` → `tests/tools/test_search.py`).

### One Tool Per Agent (invariant)

Each agent that has tool access is permitted **exactly one tool**. This is a hard rule. An agent that appears to need two tools must be split into two agents. No exceptions without explicit user authorization and an ADR update.

### Backend / Frontend Decoupling

Backend and frontend communicate only through the documented HTTP/SSE API. No server-side templating, no shared state, no tight coupling. The frontend must be replaceable without touching backend code.

### No New Dependencies Without Authorization

Do not introduce any new third-party package (runtime or dev) without:

1. Explicit user authorization.
2. A justification note added to the relevant ADR (or a new ADR if none exists).

If a task appears to require a new dependency, stop, surface the need to the user, and wait for approval before adding it to `pyproject.toml`.

### No Framework Agents

Agents are implemented directly on top of the Anthropic SDK. No LangChain, LangGraph, CrewAI, or similar framework wrappers. Orchestration logic must be explicit and readable.

### Definition of Done

A task is only complete when all of the following are true:

1. **All tests pass** — no test in the suite may be failing when work is declared done.
2. **New feature work is covered** — every net-new feature must have tests written for it. Untested features are not shippable.
3. **Bug fixes are test-first** — a bug fix must begin with a test that reproduces the bug (red), and that test must be green once the fix is applied.
4. **Final requirements review** — before closing any task, re-read the original task requirements and the relevant ADRs to confirm the implementation is consistent with both. If a conflict is found, surface it before marking done.

---

## System Agents

### Orchestrator

| Property | Value |
|---|---|
| **Model** | `claude-sonnet-4-5` |
| **Tool** | None |
| **Managed by** | `src/orchestrator.py` (not a standalone agent class — it is the coordination layer) |

**Role:**
The Orchestrator is the entry point for every task. It owns the `TaskContext`, drives the lifecycle from clarification through aggregation, and is the sole emitter of SSE events. It does not do domain work itself — it delegates everything to the Planner and executor agents.

**Responsibilities:**
- Run the clarification phase: detect ambiguities in the user's request and surface numbered questions before planning begins. Skip if the request is unambiguous.
- Invoke the Planner and validate its output before execution begins.
- Walk the task graph in dependency order; run dependency-free subtasks concurrently where possible.
- Enforce per-agent timeouts and retry budgets.
- Invoke the Aggregator once all subtasks are settled (completed or failed).
- Emit SSE events at every state transition.

**Rules:**
- Must never fabricate data. If a subtask fails, pass the failure through to the Aggregator honestly — do not synthesise a plausible result.
- Must not pass irrelevant prior agent outputs into a subtask's context. Inject only what the subtask `depends_on`.

---

### Planner

| Property | Value |
|---|---|
| **Model** | `claude-opus-4` |
| **Tool** | None (structured output only) |
| **Class** | `src/agents/planner.py` |

**Role:**
Receives the user request and any clarification answers. Produces a `TaskGraph` — a list of typed `SubTask` objects with dependency declarations. This is the only agent that reasons about the overall shape of the work.

**Output contract:**
Must return a valid `TaskGraph` (Pydantic-validated). If the response fails validation, the Orchestrator re-prompts once with the validation error appended. A second failure aborts the task with a user-facing error.

**Prompt rules:**
- Must assign each subtask to one of the registered agent types: `research`, `code`, `summary`, `aggregator`.
- Must express dependencies explicitly via `depends_on` — no implicit ordering assumptions.
- Must not invent agent types not in the registry.
- Must not produce subtasks with circular dependencies.
- Should decompose conservatively: prefer fewer, well-scoped subtasks over many fine-grained ones.
- If the request is genuinely not decomposable (e.g., a simple factual question), it is acceptable to return a single `summary` subtask.

---

### ResearchAgent

| Property | Value |
|---|---|
| **Model** | `claude-sonnet-4-5` |
| **Tool** | `search` (DuckDuckGo via `src/tools/search.py`) — **only tool permitted** |
| **Class** | `src/agents/research.py` |

**Role:**
Retrieves factual information from the web to ground subsequent agents' work. Produces a structured research summary that other agents can consume as context.

**Context received from Orchestrator:**
- `original_request`
- `clarifications`
- The subtask `description` specifying what to research

**Output:**
A prose summary of findings with source URLs inline. No fabrication — if search returns no useful results, say so explicitly rather than generating plausible-sounding facts.

**Prompt rules:**
- Must use the `search` tool at least once. Never answer a research subtask from training knowledge alone.
- Must cite sources. Any factual claim must be traceable to a search result.
- Must not use any tool other than `search`.
- If search results are irrelevant or insufficient, output a clearly labelled "insufficient data" result rather than hallucinating.

---

### CodeAgent

| Property | Value |
|---|---|
| **Model** | `claude-sonnet-4-5` |
| **Tool** | `execute_python` (subprocess sandbox via `src/tools/executor.py`) — **only tool permitted** |
| **Class** | `src/agents/code.py` |

**Role:**
Writes Python code to perform computation, data analysis, or chart generation, then executes it in a sandboxed subprocess. Returns the execution result and, where applicable, the path to any generated output file (e.g., a PNG chart).

**Context received from Orchestrator:**
- The subtask `description`
- Outputs of any `depends_on` subtasks (e.g., research data to analyse or plot)

**Output:**
```json
{
  "result": "<text summary of what the code produced>",
  "artifact_path": "<relative path to output file, or null>"
}
```

**Prompt rules:**
- Must write self-contained Python code. No assumptions about installed packages beyond the project's declared dependencies (`matplotlib`, `pandas`, `numpy`).
- Charts must be saved to the per-run output directory passed in via the tool context. Never use `plt.show()`.
- Must not use any tool other than `execute_python`.
- If execution fails (non-zero exit, exception), must inspect the stderr, fix the code, and retry — up to the Orchestrator's retry budget.
- Must not make network calls from within executed code (no `requests`, `urllib`, etc.).

---

### SummaryAgent

| Property | Value |
|---|---|
| **Model** | `claude-sonnet-4-5` |
| **Tool** | None |
| **Class** | `src/agents/summary.py` |

**Role:**
Synthesises prose from the outputs of prior agents. Used when the task requires a human-readable narrative that combines multiple sources of information.

**Context received from Orchestrator:**
- The subtask `description`
- Outputs of all `depends_on` subtasks

**Output:**
A well-structured prose summary. Length and format are guided by the subtask description.

**Prompt rules:**
- Must only synthesise from the provided context. Must not introduce facts not present in the inputs.
- Must not repeat verbatim large chunks of input — synthesise, don't transcribe.
- Must flag explicitly if the provided inputs are contradictory or insufficient to answer the subtask.

---

### AggregatorAgent

| Property | Value |
|---|---|
| **Model** | `claude-sonnet-4-5` |
| **Tool** | None |
| **Class** | `src/agents/aggregator.py` |

**Role:**
The final step in every task. Receives all agent outputs (including any failures) and produces the structured final result returned to the user.

**Context received from Orchestrator:**
- `original_request`
- `clarifications`
- All `agent_outputs` (completed and failed subtasks)

**Output contract:**
```json
{
  "answer": "<primary prose response to the user's original request>",
  "artifacts": [
    { "type": "chart", "url": "/outputs/<run_id>/<filename>.png", "caption": "..." }
  ],
  "warnings": ["<any subtask that failed or produced insufficient data>"]
}
```

**Prompt rules:**
- Must acknowledge failed subtasks in `warnings` rather than silently omitting them.
- Must not fabricate data for failed subtasks.
- The `answer` field must directly address the `original_request`, not just summarise what the agents did.
- Artifacts list must only reference files that actually exist (paths provided by CodeAgent output).
