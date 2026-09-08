# Agent application to chat with multiple LLMs

This chapter provides a LangChain agent application in Python and TypeScript for interacting with multiple LLMs (GPT, Claude, Gemini, Grok, and DeepSeek).

## Project structure

```text
chapter_4/
├── agent_app.py
├── agent_app.ts
├── package.json
├── tsconfig.json
└── README.md
```

The exercises use reusable model, CLI, and web helpers from `shared/` and `shared/essentials/`.

## Setup

Install the repository-level dependencies for your language and configure provider API keys in `shared/.env`.

## Usage

Run commands from `essentials/chapter_4`.

### Command line mode

```bash
python agent_app.py
npx tsx agent_app.ts
```

### Web API mode

```bash
python agent_app.py web
npx tsx agent_app.ts web
```

Only one web server can use the default port `8000` at a time. The API exposes service status, providers, capabilities, query, and streaming-query endpoints.
