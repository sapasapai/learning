# Text2SQL — Memory, Checkpointer & HITL Implementation Spec

> **Audience:** an AI coding agent (e.g. Copilot/Claude in VSCode).
> **Goal:** implement a backend-swappable LangGraph memory layer for an existing
> SAP HANA Text2SQL app, supporting multi-turn conversation, human-in-the-loop
> (HITL) SQL approval, and a single config flag to switch the checkpointer between
> `memory` / `hana` / `redis` / `postgres` with **no changes to graph or node code**.

---

## 0. Context (read before coding)

The application is a Retrieval-Augmented Text2SQL agent built on **LangGraph**, running
against **SAP HANA Cloud**. A knowledge base (`src/rag/kb/*.json`) describes tables,
columns, formulae (financial metrics: LCR/HQLA/NetOutflows/PercentageChange), country
code mappings, entity hierarchy/clusters, and golden SQL query pairs.

Retrieval, SQL generation, validation, and execution nodes **already exist** (or are
being built separately). **This spec covers ONLY the memory/checkpointer/HITL layer**
and the state schema those nodes read from and write to.

**Existing project layout (relevant parts):**
```
knowledgebase/
  sessions/
  src/
    agents/
    config/
    controller/
    rag/
      graph/          # LangGraph nodes + graph builder  <-- ADD state.py, followup.py, hitl.py, build.py
      kb/             # column.json, table.json, formula.json, countries.json,
                      # hierarchy.json, golden_queries.json, joins.json, rules.json, glossary.json
      vector/         # HANA vector stores (HNSW)
      router/
      tools/
      utils/
    memory/           # <-- CREATE THIS PACKAGE (checkpointer factory + config)
```

**Hard design rule:** the **only** place any checkpointer backend is named/imported is
the factory in `src/memory/checkpointer_factory.py`. Graph code compiles with whatever
the factory yields. Imports of backend packages MUST be lazy (inside the branch) so the
app runs on `memory` without redis/hana/postgres packages installed.

---

## 1. Dependencies

Add to `requirements.txt` / `pyproject.toml`. Backend-specific deps are optional extras
(only needed when that backend is selected).

```
# core (always)
langgraph>=0.2,<0.4
langgraph-checkpoint>=2.0,<3.0
langchain-core
pydantic-settings

# optional, per backend
hdbcli                          # hana
langgraph-checkpoint-hana       # hana  (BETA 0.1.0 — pin exactly; see §8 caveats)
langgraph-checkpoint-postgres   # postgres
langgraph-checkpoint-redis      # redis  (VERIFY exact package + import path against installed version)
```

> **Version coupling warning:** a checkpointer is tightly bound to the
> `langgraph-checkpoint` API version. Pin LangGraph and the checkpoint libs together.
> `langgraph-checkpoint-hana` is tested against LangGraph 0.2.x/0.3.x only.

---

## 2. Config — `src/config/memory_config.py`

Use `pydantic-settings`. All values come from environment variables. One flag
(`CHECKPOINTER_BACKEND`) selects the backend.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "memory" | "hana" | "redis" | "postgres"
    CHECKPOINTER_BACKEND: str = "memory"

    # Toggle the HITL SQL-approval step without removing it from the graph.
    ENABLE_HITL: bool = False

    # Context-window management
    TRIM_MAX_TOKENS: int = 2000

    # HANA
    HANA_HOST: str = ""
    HANA_PORT: int = 443
    HANA_USER: str = ""
    HANA_PASSWORD: str = ""

    # Redis
    REDIS_URI: str = "redis://localhost:6379"

    # Postgres
    POSTGRES_URI: str = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

memory_config = MemoryConfig()
```

`.env.example` to create:
```
CHECKPOINTER_BACKEND=memory
ENABLE_HITL=false
TRIM_MAX_TOKENS=2000
HANA_HOST=
HANA_PORT=443
HANA_USER=
HANA_PASSWORD=
REDIS_URI=redis://localhost:6379
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable
```

---

## 3. Checkpointer factory — `src/memory/checkpointer_factory.py`

**This is the central abstraction.** A context manager that yields a
`BaseCheckpointSaver`. Lazy imports per branch. Uniform `setup()` for DB backends.

```python
from contextlib import contextmanager
from typing import Iterator, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from src.config.memory_config import memory_config


@contextmanager
def get_checkpointer(backend: Optional[str] = None) -> Iterator[BaseCheckpointSaver]:
    """Yield a checkpointer for the configured backend.

    THE ONLY PLACE A CHECKPOINTER BACKEND IS NAMED OR IMPORTED.
    Switching backends = change CHECKPOINTER_BACKEND env var. No code edit.

    Usage:
        with get_checkpointer() as cp:
            graph = workflow.compile(checkpointer=cp)
            graph.invoke(..., config)   # do all graph work INSIDE the with-block
    """
    backend = (backend or memory_config.CHECKPOINTER_BACKEND).lower()

    if backend == "memory":
        # In-process only; state lost on restart. DEV/TEST ONLY — never ship.
        yield InMemorySaver()

    elif backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(memory_config.POSTGRES_URI) as cp:
            cp.setup()  # idempotent migrations
            yield cp

    elif backend == "redis":
        # VERIFY import path against the installed langgraph-checkpoint-redis version.
        from langgraph.checkpoint.redis import RedisSaver
        with RedisSaver.from_conn_string(memory_config.REDIS_URI) as cp:
            cp.setup()
            yield cp

    elif backend == "hana":
        from hdbcli import dbapi
        from langgraph_checkpoint_hana import HANASaver
        conn = dbapi.connect(
            address=memory_config.HANA_HOST,
            port=memory_config.HANA_PORT,
            user=memory_config.HANA_USER,
            password=memory_config.HANA_PASSWORD,
            encrypt=True,
        )
        cp = HANASaver(conn=conn)
        cp.setup()  # creates LANGGRAPH_CHECKPOINTS + LANGGRAPH_CHECKPOINT_WRITES if absent
        try:
            yield cp
        finally:
            conn.close()

    else:
        raise ValueError(
            f"Unknown CHECKPOINTER_BACKEND: {backend!r}. "
            f"Expected one of: memory, hana, redis, postgres."
        )
```

**Requirements / acceptance for this file:**
- Backend packages must NOT be imported at module top level (only inside branches).
- `memory` branch needs no connection and no `setup()`.
- DB branches call `setup()` exactly once.
- Unknown backend raises `ValueError` with the allowed list.
- HANA branch closes its connection in a `finally`.

> **Optional reuse note:** if the app already holds a HANA `hdbcli` connection (the
> vector stores use one), prefer injecting that shared connection into the HANA branch
> rather than opening a second. Add an optional `conn` parameter if a connection pool
> exists. Default behaviour (open + close own connection) is fine for v1.

---

## 4. State schema — `src/rag/graph/state.py`

The widened conversation state. **Every key here is persisted per `thread_id` by ANY
checkpointer.** This is what makes follow-up turns and HITL resume work.

```python
from typing import Annotated, TypedDict, Literal, Optional
from langgraph.graph.message import add_messages

RunType = Literal["ACTUAL", "FLASH"]
TurnKind = Literal["fresh", "refinement"]

class T2SQLState(TypedDict):
    # Short-term memory (chat history). add_messages reducer enables append + RemoveMessage.
    messages: Annotated[list, add_messages]

    # ---- Resolved slots: the structured conversational context ----
    # These survive across turns and are inherited on refinements.
    active_metric: Optional[str]          # e.g. "LCR" | "HQLA" | "NetOutflows" | "PercentageChange"
    active_tables: list[str]              # routed table names, scopes retrieval + few-shot
    country_codes: list[str]              # resolved ISO/entity codes, e.g. ["SG_SB"]
    run_type: RunType                     # ACTUAL (default) vs FLASH
    turn_kind: Optional[TurnKind]         # set by the follow-up classifier

    # ---- SQL lifecycle ----
    generated_sql: Optional[str]          # produced by sql_node
    last_sql: Optional[str]               # last approved/executed SQL (for "now show me X" deltas)
    validation_errors: list[str]          # from validate_node; empty = valid

    # ---- HITL ----
    hitl_decision: Optional[dict]         # populated on interrupt resume: {"action","edited_sql"?}

    # ---- Result ----
    query_result: Optional[dict]          # rows/metadata from db_node
    final_answer: Optional[str]           # answer_node output
```

**Notes for the implementer:**
- Provide a helper `initial_state(question: str) -> dict` returning the default slot
  values (`run_type="ACTUAL"`, empty lists/None) plus the first human message.
- Do NOT put resolved slots inside `messages`. They are separate keys precisely so that
  message-trimming (§6) never drops conversational context.

---

## 5. Follow-up classifier + slot inheritance — `src/rag/graph/followup.py`

Decides whether a new turn is a **fresh** question or a **refinement** of the prior one,
and resets or inherits slots accordingly.

```python
from src.rag.graph.state import T2SQLState

REFINEMENT_HINTS = [
    "now", "instead", "what about", "and ", "just ", "only ",
    "change", "switch", "also", "that", "those", "same",
]

def _looks_like_refinement(text: str, state: T2SQLState) -> bool:
    """Cheap heuristic prefilter. If a metric is already active and the new
    message contains a refinement cue and names no new metric, treat as refinement."""
    t = text.lower().strip()
    has_active = state.get("active_metric") is not None
    cue = any(h in t for h in REFINEMENT_HINTS)
    return has_active and cue

def classify_and_inherit(state: T2SQLState) -> dict:
    """Node. Sets turn_kind. On 'fresh', clears slots. On 'refinement', keeps them
    (downstream nodes override only the delta the user mentioned)."""
    user_msg = state["messages"][-1].content

    # Optionally replace _looks_like_refinement with a small LLM classifier for
    # robustness. Keep it cheap; it runs every turn. The LLM prompt should return
    # strictly "fresh" or "refinement" given (prior slots, new message).
    is_refine = _looks_like_refinement(user_msg, state)

    if not is_refine:
        return {
            "turn_kind": "fresh",
            "active_metric": None,
            "active_tables": [],
            "country_codes": [],
            "run_type": "ACTUAL",
            "generated_sql": None,
            "validation_errors": [],
        }

    # Refinement: keep prior slots; just tag the turn. Downstream nodes (geography,
    # metric, router) read current state as defaults and overwrite only what changed.
    return {"turn_kind": "refinement"}
```

**Slot-override contract for downstream nodes (geography / metric / router):**
Each MUST read the current slot value as its default and only overwrite when the
new user message explicitly changes it. Example behaviours to implement/verify:

| Turn | User says | Override | Inherit (unchanged) |
|------|-----------|----------|---------------------|
| 1 | "LCR for Group Conso" | metric=LCR, tables=[…], codes=[expanded cluster], run=ACTUAL | — |
| 2 | "now just Singapore" | country_codes=["SG_SB"] | metric=LCR, run=ACTUAL, tables |
| 3 | "as flash instead" | run_type="FLASH" | metric=LCR, country_codes, tables |

> The classifier only resets/keeps slots. The actual override logic lives in the
> existing geography/metric/router nodes — they must honour this contract.

---

## 6. Context-window management — trim node (`src/rag/graph/trim.py`)

Long conversations exceed the model context window. Trim **messages only**; never trim
the resolved slots.

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from src.config.memory_config import memory_config
from src.rag.graph.state import T2SQLState

def trim_node(state: T2SQLState) -> dict:
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=memory_config.TRIM_MAX_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": trimmed}
```

Place `trim_node` right after `classify_and_inherit` (so classification sees full
history, then we trim before the LLM-heavy nodes). Summarization is out of scope for v1;
add later only if sessions get very long.

---

## 7. Human-in-the-loop SQL approval — `src/rag/graph/hitl.py`

HITL uses LangGraph `interrupt()`. The graph pauses, state is checkpointed, the human
reviews the generated SQL, then resumes via `Command(resume=...)` on the **same
`thread_id`**. Works on every backend; only durable backends survive process restart.

```python
from langgraph.types import interrupt
from src.rag.graph.state import T2SQLState

def hitl_review_node(state: T2SQLState) -> dict:
    """Pause for human approval of generated SQL. Resumes with a decision dict:
        {"action": "approve"}
        {"action": "edit", "edited_sql": "<sql>"}
        {"action": "reject"}
    """
    decision = interrupt({
        "type": "sql_approval",
        "generated_sql": state.get("generated_sql"),
        "tables": state.get("active_tables"),
        "metric": state.get("active_metric"),
        "country_codes": state.get("country_codes"),
        "run_type": state.get("run_type"),
        "validation_errors": state.get("validation_errors", []),
        "prompt": "Review and approve this SQL before execution.",
    })

    action = (decision or {}).get("action")
    if action == "approve":
        return {"hitl_decision": decision, "last_sql": state.get("generated_sql")}
    if action == "edit":
        edited = decision.get("edited_sql")
        return {"hitl_decision": decision, "generated_sql": edited, "last_sql": edited}
    # reject (or unknown) -> clear SQL so router sends it back to regeneration
    return {"hitl_decision": decision, "generated_sql": None}
```

**Routing after HITL (conditional edges):**
```python
def route_after_hitl(state: T2SQLState) -> str:
    decision = state.get("hitl_decision") or {}
    action = decision.get("action")
    if action == "approve":
        return "db"            # execute
    if action == "edit":
        return "validate"      # re-validate the human's edited SQL, then back to hitl
    return "sql"               # reject -> regenerate
```

**HITL toggle:** when `ENABLE_HITL` is false, the graph must route `validate -> db`
directly (skip `hitl_review`). Implement this with a conditional edge that reads
`memory_config.ENABLE_HITL`, OR build two edge sets at graph-construction time. Prefer
construction-time branching so the disabled path has zero interrupt overhead.

---

## 8. Graph assembly — `src/rag/graph/build.py`

```python
from langgraph.graph import StateGraph, START, END
from src.config.memory_config import memory_config
from src.rag.graph.state import T2SQLState
from src.rag.graph.followup import classify_and_inherit
from src.rag.graph.trim import trim_node
from src.rag.graph.hitl import hitl_review_node, route_after_hitl
# existing nodes (imported from wherever they live):
# geography_node, metric_node, table_router_node, column_retrieval_node,
# example_retrieval_node, sql_node, validate_node, db_node, answer_node

def build_workflow() -> StateGraph:
    wf = StateGraph(T2SQLState)

    wf.add_node("classify", classify_and_inherit)
    wf.add_node("trim", trim_node)
    wf.add_node("geography", geography_node)
    wf.add_node("metric", metric_node)
    wf.add_node("router", table_router_node)
    wf.add_node("retrieve", column_retrieval_node)
    wf.add_node("examples", example_retrieval_node)
    wf.add_node("sql", sql_node)
    wf.add_node("validate", validate_node)
    wf.add_node("db", db_node)
    wf.add_node("answer", answer_node)

    wf.add_edge(START, "classify")
    wf.add_edge("classify", "trim")
    wf.add_edge("trim", "geography")
    wf.add_edge("geography", "metric")
    wf.add_edge("metric", "router")
    wf.add_edge("router", "retrieve")
    wf.add_edge("retrieve", "examples")
    wf.add_edge("examples", "sql")
    wf.add_edge("sql", "validate")

    if memory_config.ENABLE_HITL:
        wf.add_node("hitl_review", hitl_review_node)
        # validate -> hitl (if valid) ; validate -> sql (if errors) handled in validate's router
        wf.add_edge("validate", "hitl_review")
        wf.add_conditional_edges("hitl_review", route_after_hitl,
                                 {"db": "db", "validate": "validate", "sql": "sql"})
    else:
        wf.add_edge("validate", "db")

    wf.add_edge("db", "answer")
    wf.add_edge("answer", END)
    return wf
```

> If `validate_node` already routes back to `sql` on errors, keep that conditional edge
> and only attach the HITL edges on the success path. Adjust to the existing validate
> contract — do not duplicate routing.

---

## 9. Runner / entry point — `src/rag/graph/run.py`

```python
from langgraph.types import Command
from src.memory.checkpointer_factory import get_checkpointer
from src.rag.graph.build import build_workflow
from src.rag.graph.state import initial_state  # helper from §4

def run_turn(question: str, thread_id: str) -> dict:
    """Run a single conversational turn. Same thread_id continues a conversation."""
    wf = build_workflow()
    with get_checkpointer() as cp:                 # backend chosen by env flag
        graph = wf.compile(checkpointer=cp)        # identical for every backend
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke(initial_state(question), config)

def resume_turn(thread_id: str, decision: dict) -> dict:
    """Resume a HITL-interrupted turn with the human's decision."""
    wf = build_workflow()
    with get_checkpointer() as cp:
        graph = wf.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke(Command(resume=decision), config)

def get_thread_state(thread_id: str):
    wf = build_workflow()
    with get_checkpointer() as cp:
        graph = wf.compile(checkpointer=cp)
        return graph.get_state({"configurable": {"thread_id": thread_id}})

def delete_thread(thread_id: str) -> None:
    with get_checkpointer() as cp:
        cp.delete_thread(thread_id)
```

**Important runtime rules:**
- All graph work happens INSIDE the `with get_checkpointer()` block (DB backends are
  context managers; the connection must be open during invoke).
- `run_turn` returns either a normal result OR an interrupt payload (when HITL pauses).
  Detect interrupts in the caller: check `result.get("__interrupt__")` (LangGraph
  surfaces interrupts on the returned state); if present, surface the payload to the
  human and later call `resume_turn` with the same `thread_id`.
- Resume MUST use the same `thread_id`. On durable backends this works across process
  restarts and across workers; on `memory` only within the same process.

---

## 10. Test plan (implement these; gate the PR on them)

Use `pytest`. All tests run on `CHECKPOINTER_BACKEND=memory` (no external services).

**Multi-turn / state inheritance**
1. Fresh question sets `turn_kind="fresh"` and populates slots.
2. Refinement ("now just Singapore") sets `turn_kind="refinement"`, overrides
   `country_codes`, leaves `active_metric` and `run_type` unchanged.
3. Refinement ("as flash instead") flips `run_type` to FLASH, keeps metric + codes.
4. State for two different `thread_id`s does not leak between threads.

**Trim**
5. After exceeding `TRIM_MAX_TOKENS`, `messages` is trimmed but resolved slots persist.

**HITL (ENABLE_HITL=true)**
6. Graph pauses at `hitl_review`; returned state exposes the interrupt payload with the
   generated SQL.
7. `resume_turn(thread_id, {"action":"approve"})` continues to `db` and produces a result.
8. `{"action":"edit","edited_sql":...}` routes to `validate` then back, and the edited
   SQL is what reaches `db`.
9. `{"action":"reject"}` clears `generated_sql` and routes back to `sql`.
10. `ENABLE_HITL=false` routes `validate -> db` with no interrupt.

**Factory**
11. Unknown backend raises `ValueError`.
12. `memory` backend requires no network and no `setup()`.
13. (Integration, optional/skippable) each DB backend's `setup()` creates tables and a
    round-trip checkpoint survives a new `get_checkpointer()` context.

**Backend-swap invariance**
14. The SAME graph + node code, run under `memory` then (mocked) durable backend,
    produces identical state transitions for the multi-turn sequence above.

---

## 11. Backend caveats (bake into code comments + README)

- **memory** (`InMemorySaver`): in-process only, lost on restart. Dev/test default.
  Never ship to production. No `setup()`, no connection.
- **postgres** (`langgraph-checkpoint-postgres`): most mature DB option. `from_conn_string`
  is a context manager; `setup()` runs migrations. Recommended default for production
  unless state must live in HANA.
- **redis** (`langgraph-checkpoint-redis`): VERIFY exact package name and import path
  against the installed version — Redis checkpointer packaging has shifted. Lazy import
  means a wrong path only fails when the redis branch is selected.
- **hana** (`langgraph-checkpoint-hana`): BETA 0.1.0, single maintainer, tested only
  against LangGraph 0.2.x/0.3.x and langgraph-checkpoint 2.x. PIN THE EXACT VERSION.
  `hdbcli` is synchronous; async checkpointer methods delegate to sync — for
  high-concurrency async deployments wrap calls in `asyncio.to_thread()`. Creates
  `LANGGRAPH_CHECKPOINTS` and `LANGGRAPH_CHECKPOINT_WRITES` (NCLOB columns, keyed by
  `thread_id, checkpoint_ns, checkpoint_id`). Choose this ONLY if there is a real
  requirement that agent state stay co-located in HANA (audit/compliance). Read its
  source before adopting.

## 12. Migrations / ops

- Run DB `setup()` as a dedicated deployment step (or at controlled startup), not on
  every request. The factory calls `setup()` per context for simplicity; for high-QPS
  deployments, move `setup()` to a one-time init script and remove it from the hot path.
- Connection reuse: in production wire the HANA checkpointer to the shared connection
  pool rather than opening a connection per turn.

## 13. Definition of done

- [ ] `src/config/memory_config.py` + `.env.example`
- [ ] `src/memory/checkpointer_factory.py` (lazy imports, 4 backends, ValueError on unknown)
- [ ] `src/rag/graph/state.py` (`T2SQLState` + `initial_state`)
- [ ] `src/rag/graph/followup.py` (classifier + inheritance contract)
- [ ] `src/rag/graph/trim.py`
- [ ] `src/rag/graph/hitl.py` (interrupt node + router) honouring `ENABLE_HITL`
- [ ] `src/rag/graph/build.py` (graph wired, HITL toggle at construction time)
- [ ] `src/rag/graph/run.py` (run / resume / get_state / delete_thread)
- [ ] pytest suite from §10 passing on `memory` backend
- [ ] README section documenting the env flag and how to switch backends
