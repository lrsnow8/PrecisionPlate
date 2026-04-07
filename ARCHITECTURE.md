# PrecisionPlate — Architecture

## Overview

PrecisionPlate is a conversational AI nutrition assistant built as a single LangGraph agent. Users interact with it via natural language — logging meals, checking progress, setting goals, and getting recommendations — as if talking to a personal nutritionist. The agent manages all state, memory, and tool execution internally.

---

## Technology Stack

| Layer | Technology |
|---|---|
| LLM | `claude-sonnet-4-6` via `langchain-anthropic` |
| Agent Framework | LangGraph `StateGraph` + `ToolNode` |
| Conversation Persistence | LangGraph `SqliteSaver` checkpointer |
| RAG Vector Store | ChromaDB (local, persistent on disk) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Meal Log Database | SQLite via Python `sqlite3` |
| Image Vision | Existing multimodal code (`image_to_macro/image_to_macro.py`) |
| CLI Interface | `rich` + `prompt_toolkit` |

---

## Directory Structure

```
PrecisionPlate/
├── main.py                          # Entry point — starts the CLI chat loop
├── ARCHITECTURE.md                  # This file
├── requirements.txt                 # All pinned dependencies
│
├── agent/
│   ├── graph.py                     # Builds the LangGraph StateGraph
│   ├── state.py                     # NutritionState TypedDict definition
│   └── prompts.py                   # System prompt / nutritionist persona
│
├── tools/
│   ├── meal_logger.py               # Log a meal from text description
│   ├── meal_logger_vision.py        # Log a meal from a photo (uses image_to_macro)
│   ├── nutrition_lookup.py          # RAG query against ChromaDB nutrition docs
│   ├── daily_summary.py             # Query SQLite for today's macro totals
│   ├── goal_manager.py              # Set / read user macro and calorie goals
│   ├── meal_recommendations.py      # Suggest next meal based on remaining macros + RAG
│   └── historical_report.py         # Weekly / historical aggregation from SQLite
│
├── rag/
│   ├── ingest.py                    # One-time script: chunk, embed, load docs into Chroma
│   ├── retriever.py                 # Query ChromaDB, return top-k relevant chunks
│   └── docs/                        # Raw nutrition knowledge base (text / PDF)
│       ├── usda_dietary_guidelines.txt
│       └── ...
│
├── db/
│   ├── database.py                  # Schema creation, connection helper, CRUD functions
│   └── precision_plate.db           # Auto-created SQLite file (gitignored)
│
└── image_to_macro/
    └── image_to_macro.py            # Existing vision code (reused by meal_logger_vision)
```

---

## SQLite Schema

Managed by `db/database.py`. The LangGraph `SqliteSaver` checkpointer creates its own internal tables in the same DB file for conversation checkpointing.

```sql
-- User profile
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Daily macro / calorie goals
CREATE TABLE goals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),
    calories    REAL,
    protein_g   REAL,
    carbs_g     REAL,
    fat_g       REAL,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Meal sessions (a single eating event)
CREATE TABLE meals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER REFERENCES users(id),
    logged_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    description  TEXT,
    source       TEXT   -- 'text' or 'photo'
);

-- Individual food items within a meal
CREATE TABLE meal_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    meal_id     INTEGER REFERENCES meals(id),
    food_name   TEXT,
    calories    REAL,
    protein_g   REAL,
    carbs_g     REAL,
    fat_g       REAL
);
```

> **Conversation history and checkpoints** are stored automatically by the `SqliteSaver` checkpointer — no manual schema required for that.

---

## LangGraph Agent Design

### State

```python
# agent/state.py
class NutritionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation (managed by checkpointer)
    summary: str          # compressed long-term memory summary
    user_id: str          # identifies the user across sessions
    today_snapshot: dict  # today's macro totals, refreshed each turn
```

### Graph Structure

```
┌─────────────┐
│  __start__  │
└──────┬──────┘
       │
┌──────▼──────────────┐
│  load_context node  │  ← injects today's macro snapshot into state
└──────┬──────────────┘
       │
┌──────▼──────┐
│   chatbot   │  ← Claude with all tools bound; reads summary + messages
└──────┬──────┘
       │
  tool calls?
  /          \
yes           no
 │             │
┌▼──────────┐ ┌▼──────────┐
│  tools    │ │  should   │
│  node     │ │ summarize │
│  (all 7)  │ │    ?      │
└─────┬─────┘ └──┬─────┬──┘
      │        yes     no
      │          │      │
      │   ┌──────▼──┐   │
      │   │summarize│   │
      │   │  node   │   │
      │   └──────┬──┘   │
      │          │      │
      └──► chatbot ◄────┘
                 │
           ┌─────▼──────┐
           │  __end__   │
           └────────────┘
```

### Nodes

| Node | Responsibility |
|---|---|
| `load_context` | Runs on every turn; queries SQLite for today's macro totals and injects into state |
| `chatbot` | Calls Claude with the system prompt (persona + summary + today snapshot) + full message history; decides which tools to call |
| `tools` | LangGraph `ToolNode`; executes whichever tool(s) Claude chose and returns results |
| `summarize` | Fires when `len(messages) > 20`; calls Claude to compress older messages into `state.summary`; trims raw message list to last 10 |

### Routing Logic

```python
def should_continue(state: NutritionState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    if len(state["messages"]) > 20:
        return "summarize"
    return END
```

### Persistence (Cross-Session Memory)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("db/precision_plate.db")
graph = graph_builder.compile(checkpointer=checkpointer)

# Each user gets a unique thread_id — history is automatically restored on next run
config = {"configurable": {"thread_id": user_id}}
graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
```

The `SqliteSaver` checkpointer persists the entire graph state (messages + summary + snapshot) after every node execution. On the next session, invoking with the same `thread_id` automatically restores all prior state — no manual load/save logic required.

---

## RAG Pipeline

### Ingestion (one-time, run `rag/ingest.py`)

```
Raw docs (USDA guidelines, WHO nutrition docs, dietary science papers)
  → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
  → sentence-transformers/all-MiniLM-L6-v2 embeddings
  → ChromaDB collection: "nutrition_knowledge" (persisted to disk)
```

### Retrieval (at agent runtime via `nutrition_lookup` tool)

```
User query or agent sub-query
  → embed with same sentence-transformers model
  → ChromaDB cosine similarity search (top 5 chunks)
  → returned as context string injected into tool result
```

### Tools that use RAG

- `nutrition_lookup` — direct knowledge queries ("how much fiber should I eat?")
- `meal_recommendations` — combines remaining macro budget (from SQLite) with RAG-retrieved dietary guidance to suggest a specific next meal

---

## Tool Definitions

All tools are decorated with LangChain's `@tool` and registered with the `chatbot` node.

| Tool | Input | Output |
|---|---|---|
| `log_meal_text` | Natural language meal description | Parsed macros stored in SQLite; confirmation string |
| `log_meal_photo` | File path to image | Vision-extracted macros stored in SQLite; confirmation string |
| `get_daily_summary` | None (uses `user_id` from state) | Today's calories + macros consumed vs. goal |
| `get_nutrition_info` | Free-text query | Top-k RAG chunks from ChromaDB as a string |
| `set_goal` | calories, protein_g, carbs_g, fat_g | Confirmation of updated goal in SQLite |
| `get_meal_recommendation` | None | Suggested meal based on remaining macros + RAG guidance |
| `get_historical_report` | Period (`"week"`, `"month"`) | Aggregated meal stats from SQLite |

---

## Summarization Memory Flow

```
Turn N:  messages = [m1, m2, ..., m21]  → triggers summarize node
         Claude compresses m1..m11 → summary string
         state.summary = "<compressed history>"
         state.messages = [m12, ..., m21]  (last 10 retained as raw)

Turn N+1: system prompt contains:
          [LONG-TERM MEMORY]
          <summary>

          [RECENT CONVERSATION]
          m12 ... m21

          [TODAY'S NUTRITION SNAPSHOT]
          <from load_context node>
```

---

## CLI → Web Adaptability

The CLI loop in `main.py` is a thin wrapper over `graph.invoke()`. Migrating to a web UI requires only:

1. Add `FastAPI` with a `POST /chat` endpoint that calls `graph.invoke()`
2. Use `thread_id = session_id` (from HTTP session or JWT) as the checkpointer key
3. Add a simple HTML/JS frontend or React SPA
4. **Zero changes** to agent, tools, RAG, or database code

---

## Build Order

1. `requirements.txt` — pin all dependencies
2. `db/database.py` — schema creation + CRUD helpers
3. `rag/ingest.py` + `rag/retriever.py` — ChromaDB pipeline
4. `tools/*.py` — all 7 LangChain tools
5. `agent/state.py` — `NutritionState` TypedDict
6. `agent/prompts.py` — nutritionist system prompt
7. `agent/graph.py` — LangGraph `StateGraph` wiring all nodes
8. `main.py` — CLI loop
