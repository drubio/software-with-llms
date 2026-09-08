# Agent application with selective memory retrieval using BM25

This chapter upgrades the Chapter 5 memory agent from full-history inclusion to retrieval-based prompt inclusion. It uses BM25 and lexical-overlap gating to select relevant prior turns while preserving persistent session memory and structured responses.

## Project structure

```text
chapter_6/
├── agent_memory_retrieval.py
├── agent_memory_retrieval.ts
├── package.json
├── tsconfig.json
└── README.md
```

## Usage

Run commands from `essentials/chapter_6`.

```bash
python agent_memory_retrieval.py
npx tsx agent_memory_retrieval.ts
```

Add `web` to either command to run the web API variant.

Successful responses include retrieval diagnostics such as available and retrieved message counts, token estimates, and estimated token savings. Session memory remains isolated by provider and session ID.
