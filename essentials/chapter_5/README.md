# Agent application with memory and persistent chat

This chapter extends Chapter 4 with persistent session memory and structured output parsing. Both exercises are available in Python and TypeScript and support CLI and web API modes.

## Project structure

```text
chapter_5/
├── agent_memory_persist.py
├── agent_memory_persist.ts
├── agent_structured_output.py
├── agent_structured_output.ts
├── package.json
├── tsconfig.json
└── README.md
```

The chapter builds directly on `chapter_4/agent_app.py` and `chapter_4/agent_app.ts`, plus the centralized helpers in `shared/`.

## Usage

Run commands from `essentials/chapter_5`.

```bash
python agent_memory_persist.py
python agent_structured_output.py
npx tsx agent_memory_persist.ts
npx tsx agent_structured_output.ts
```

Add `web` to any command to run the web API variant.

## Learning goals

- **Memory and persistence** demonstrates reusable conversation state across runs.
- **Structured output** keeps memory while returning stable fields such as `answer`, `summary`, `keywords`, and `distilled`.
- Memory-capable APIs add `/history` and `/reset-memory` endpoints to the Chapter 4 API.
