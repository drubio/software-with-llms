# Agent application with tools

This chapter extends Chapter 6's features with **tool calling**.

It provides cross-framework (**LangChain**, **LlamaIndex**) and dual-language (**Python**, **TypeScript**) implementations that:

- Reuse Chapter 4 base provider/client setup and shared CLI/Web helpers
- Reuse Chapter 5 memory + persistence base behavior through Chapter 6 managers
- Reuse Chapter 6 retrieval-memory + structured-response pattern
- Add Chapter 7 tool orchestration (model decides tool call, tool executes, model synthesizes final answer)

Just like earlier chapters, each script can run in:
- **Command line mode**
- **Web API mode**

## Project structure

```text
chapter_7/
├── langchain/
│   ├── agent_tools.py
│   └── agent_tools.ts
├── llamaindex/
│   ├── agent_tools.py
│   └── agent_tools.ts
├── tools.py
├── tools.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Script matrix

| Framework | Python | TypeScript |
|---|---|---|
| **LangChain** | `langchain/agent_tools.py` | `langchain/agent_tools.ts` |
| **LlamaIndex** | `llamaindex/agent_tools.py` | `llamaindex/agent_tools.ts` |

## Dependencies and environment

Chapter 7 builds on Chapter 4/5/6:

- Chapter 4 shared utilities and web server helpers
- Chapter 5 memory + persistence foundations
- Chapter 6 retrieval-memory manager classes
- Chapter 7 tool utilities (`tools.py`, `tools.ts`)

Set API keys in `shared/.env`:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
```

Install the repository-level Node dependencies for the TypeScript examples:

```bash
npm run install:root
```

## Usage

Run commands from `essentials/chapter_7`.

### Command line mode

#### Python

```bash
python langchain/agent_tools.py
python llamaindex/agent_tools.py
```

#### TypeScript

```bash
npx tsx langchain/agent_tools.ts
npx tsx llamaindex/agent_tools.ts
```

### Web API mode

```bash
python langchain/agent_tools.py web
python llamaindex/agent_tools.py web
npx tsx langchain/agent_tools.ts web
npx tsx llamaindex/agent_tools.ts web
```

## Tool orchestration pattern

All Chapter 7 agents follow the same two-step JSON tool loop:

1. Model returns JSON with:
   - `tool_calls`: an array of tool calls, each shaped like `{ "name": "...", "arguments": { ... }, "output": null }`
   - `final_answer`: short draft answer
2. Agent executes tools locally when `tool_calls` entries are present.
3. Agent asks model for a final JSON response that keeps the same `tool_calls` array and fills in each `output`.

### Expected response shape

```json
{
  "tool_calls": [{
    "name": "tool_name",
    "arguments": {"arg": "value"},
    "output": "serialized tool output"
  }],
  "final_answer": "..."
}
```

- If no tool is needed, `tool_calls` is an empty array.
- On success, `raw_answer`/`rawAnswer` mirrors the final answer text.

## Included tool utilities

Current tools shared by Python/TypeScript utilities:

- `get_wikipedia_evidence_pack` — fetch Wikipedia summary + references + Wikimedia media

Both utility modules expose:

- tool definitions metadata (`TOOL_DEFINITIONS`)
- a dispatcher (`run_tool` in Python / `runTool` in TypeScript)
- a prompt helper (`build_tools_prompt` in Python / `buildToolsPrompt` in TypeScript)

## Notes

- Tool contracts are intentionally simple and framework-agnostic for easy extension in later chapters.
- Session memory compatibility and provider handling continue to come from inherited chapter managers.
- If a model returns non-JSON output, the agent surfaces a parsing error with the raw response context.
