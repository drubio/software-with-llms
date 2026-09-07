# Agent application with memory and persistent chat for multiple LLMs

This chapter extends Chapter 4's Agent application with:

1. **Memory and persistence**
2. **Memory, persistence and structured output parsing**

Both variations are available across **LangChain** and **LlamaIndex**, in **Python** and **TypeScript**, as well as CLI and web API modes.

## Project structure

```text
chapter_5/
├── langchain/
│   ├── agent_memory_persist.py
│   ├── agent_memory_persist.ts
│   ├── agent_structured_output.py
│   └── agent_structured_output.ts
├── llamaindex/
│   ├── agent_memory_persist.py
│   ├── agent_memory_persist.ts
│   ├── agent_structured_output.py
│   └── agent_structured_output.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Script matrix

| Framework | Memory and Persistence (Python) | Memory and Persistence (TypeScript) | Structured Output (Python) | Structured Output (TypeScript) |
|---|---|---|---|---|
| **LangChain** | `langchain/agent_memory_persist.py` | `langchain/agent_memory_persist.ts` | `langchain/agent_structured_output.py` | `langchain/agent_structured_output.ts` |
| **LlamaIndex** | `llamaindex/agent_memory_persist.py` | `llamaindex/agent_memory_persist.ts` | `llamaindex/agent_structured_output.py` | `llamaindex/agent_structured_output.ts` |

## Dependencies and environment

Chapter 5 reuses Chapter 4 framework managers plus centralized shared components:
- `chapter_4/langchain/agent_app.py` / `chapter_4/langchain/agent_app.ts`
- `chapter_4/llamaindex/agent_app.py` / `chapter_4/llamaindex/agent_app.ts`
- `shared/llm_models.py` / `shared/llm_models.ts`
- `shared/utils.py` / `shared/utils.ts` and `shared/web.py` / `shared/web.ts`
- `shared/essentials/utils.py` / `shared/essentials/utils.ts`
- `shared/essentials/web.py` / `shared/essentials/web.ts`

Ensure you install the dependencies for your language of choice located in the root level folder—requirements.txt or package.json—in addition to declaring LLM API keys in the shared/.env file. See the shared/ folder README.md for additional details.


## Usage

Run commands from `essentials/chapter_5`. Install the repository-level Node dependencies with `npm run install:root` before running TypeScript examples.

### Command line mode

#### Python

```bash
python langchain/agent_memory_persist.py
python langchain/agent_structured_output.py
python llamaindex/agent_memory_persist.py
python llamaindex/agent_structured_output.py
```

#### TypeScript

```bash
npx tsx langchain/agent_memory_persist.ts
npx tsx langchain/agent_structured_output.ts
npx tsx llamaindex/agent_memory_persist.ts
npx tsx llamaindex/agent_structured_output.ts
```

### Web API mode

```bash
python langchain/agent_memory_persist.py web
npx tsx llamaindex/agent_structured_output.ts web
```

## Incremental learning goal

- **Memory and Persistence** shows reusable conversation state across runs.
- **Structured Outputs** keep memory while adding JSON output parsing, so downstream code can consume stable fields (`answer`, `summary`, `keywords`, `distilled`).

## Memory-aware endpoints

In addition to base Chapter 4 endpoints (`/`, `/providers`, `/query`, `/health`), memory-capable managers expose:

| Method | Path | Description |
|---|---|---|
| GET | `/history?session_id=<id>` | Get stored turns for a session |
| POST | `/reset-memory` | Clear memory by session or clear all |

### Query with session context

Use `session_id` in `/query` requests so consecutive calls share context:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "Twinkle, Twinkle, Little",
        "provider": "openai",
        "session_id": "default"
      }'
```

### Read session history

```bash
curl "http://localhost:8000/history?session_id=default"
```

### Reset memory

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
  -d '{"session_id": "default"}'
```

To clear all sessions use no parameters:

```bash
curl -X POST http://localhost:8000/reset-memory \
  -H "Content-Type: application/json" \
```


## Notes

- Session memory is isolated by `session_id`.
- Persistent sessions are stored under each framework's `sessions/` directory.
- Structured variants return parsed JSON in `response` and keep a short `raw_answer`/`rawAnswer` field.
