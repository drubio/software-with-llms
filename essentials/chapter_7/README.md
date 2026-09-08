# Agent application with tools

This chapter extends Chapter 6's retrieval and memory features with LangChain tool calling. The model can select a local tool, the agent executes it, and the model synthesizes a final structured answer.

## Project structure

```text
chapter_7/
├── agent_tools.py
├── agent_tools.ts
├── tools.py
├── tools.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Usage

Run commands from `essentials/chapter_7`.

```bash
python agent_tools.py
npx tsx agent_tools.ts
```

Add `web` to either command to run the web API variant.

## Tool orchestration

1. The model returns a JSON response containing `tool_calls` and `final_answer`.
2. The agent executes requested tools locally.
3. The model produces a final JSON response with each tool output included.

The included Wikipedia tools are defined in `tools.py` and `tools.ts`.
