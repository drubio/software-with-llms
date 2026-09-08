# Chapter 8 — Unified LLM UI App (Streaming + Standard Response Modes)

This chapter is the final UI showcase for the Agent Essentials book.

It contains **one Next.js app** that demonstrates the same backend-powered LLM interactions through **3 different UI component approaches**:

1. **LangChain Chat UI** (`@langchain/langgraph-sdk/react`)
2. **Assistant UI** (`@assistant-ui/react`)
3. **Custom Chat UI** (vanilla React implementation)

All three views share the same runtime settings sidebar and can talk to the earlier chapter backends.

---

## What this chapter demonstrates

### 1) Multiple UI implementations for similar chat behavior
Each framework tab exposes similar behavior (provider selection, temperature, max tokens, optional memory/history) so you can compare integration style and developer ergonomics.

### 2) **Standard (non-streaming)** vs **Streaming** responses
The app supports two response modes from the settings panel:

- **Streaming**: Progressive chunk rendering when backend streaming is available.
- **Standard**: Classic request/response (render full answer when complete).

If streaming is unavailable, the Streaming option is disabled and the app falls back to Standard mode.

### 3) Backend capability detection
The UI checks backend status and capabilities to decide whether streaming is available (similar to the online/offline indicator pattern).

---

## Architecture overview

### Frontend (this chapter)
- Next.js app in `essentials/chapter_8`
- Main UI in `app/page.tsx`

### Backend (earlier chapters)
This UI expects a backend on `http://localhost:8000` with:

- `GET /` (status)
- `GET /providers`
- `POST /query` (standard single provider)
- `POST /query-all` (standard multi-provider)
- `GET /capabilities` (feature discovery, returns `streaming: true|false` and `coagent: true|false`)
- `POST /query-stream` (streaming endpoint for progressive output)
- Optional memory endpoints:
  - `GET /history`
  - `POST /reset-memory`

Streaming helpers are provided by the shared Python and TypeScript web modules used by the backend chapters.

---

## Prerequisites

- Node.js 18+
- npm
- A running backend web API on port `8000`
- Provider API keys configured according to earlier chapters (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

---

## Running the app

## 1) Start the backend (example)
From a backend chapter that uses the shared web API (for example Chapter 7):

```bash
cd essentials/chapter_7
# Python example
python agent_tools.py web

# or TypeScript example
npx tsx agent_tools.ts web
```

You should have an API available at `http://localhost:8000`.

## 2) Install frontend deps

```bash
cd essentials/chapter_8
npm install
```

NOTE: This chapter does **not** use the repository-level shared libraries in `shared/`

## 3) Start Next.js

```bash
npm run dev
```

Open `http://localhost:3000`.

---

## Using the UI

1. Pick one of the 3 UI tabs at the top.
2. Open settings (gear icon).
3. Configure:
   - Query Mode: single provider or all providers
   - Provider (in single mode, defaults to OpenAI when available)
   - Response Mode: streaming / standard
   - Temperature / max tokens
   - Session ID (if memory-enabled backend)
4. Send prompts and compare behavior across frameworks.

### Streaming behavior notes
- Streaming is primarily used in **single-provider mode**.
- If streaming is unavailable, choose **Standard** mode.
- The app defaults to **Streaming** when backend capabilities indicate support.

---

## Troubleshooting

- **API Offline in UI**: verify backend is running on `localhost:8000`.
- **No providers shown**: verify API keys are set for at least one provider.
- **Streaming option disabled**: backend may not implement `/capabilities` with `streaming: true`.
- **Co-agent sidebar**: it only appears when the backend reports `coagent: true` in `/capabilities`
- **Memory buttons unavailable**: backend manager may not support memory/history.

---

## Goal of this chapter

This chapter is intentionally not about one “best” chat component.
It is a side-by-side comparison showing how different UI stacks can integrate with the same LLM backend while supporting both:

- **Synchronous request/response UX**, and
- **Asynchronous streamed UX**.
