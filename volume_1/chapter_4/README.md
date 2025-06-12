# LLM Gateway

This chapter provides a cross-framework (Langchain, LlamaIndex), dual-language (Python, JavaScript) gateway to interact with various LLMs (GPT, Claude, Gemini, Grok)

It can be run in two forms:
- As a **command line** script for console exploration.
- As a **web API** server for HTTP access through a web frontend or other web tool. 

## Project structure and characteristics
```
chapter_4/
├── langchain/
│   ├── llm_gateway.py
│   └── llm_gateway.js
├── llamaindex/
│   ├── llm_gateway.py
│   └── llm_gateway.js
├── utils.py
├── utils.js
├── web.py
├── web.js
├── requirements.txt
├── package.json
└── .env
````

You choose what framework to run by choosing the script language and framework subfolder:

------------------------------------------------------------------------------------
| Framework       | Python                          | JavaScript                   |
|-----------------|---------------------------------|------------------------------|
| **LangChain**   | `langchain/llm_gateway.py`   | `langchain/llm_gateway.js`      |
| **LlamaIndex**  | `llamaindex/llm_gateway.py`  | `llamaindex/llm_gateway.js`     |
------------------------------------------------------------------------------------

The following files are **shared** for both Langchain and LlamaIndex implementations:

- `utils.py` / `utils.js` — shared CLI and logic
- `web.py` / `web.js` — shared web server for all frameworks
- `requirements.txt` / `package.json` — shared package dependencies
- `.env` - Environment file with LLM API keys

## Environment Setup
Update the `.env` file with the LLM API keys:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
````

## Usage Modes

Run in either:

* [Command Line Mode]
### ❯ Example (Python)
```bash
python langchain/llm_gateway.py
python llamaindex/llm_gateway.py
```

### ❯ Example (JavaScript)

```bash
node langchain/llm_gateway.js
node llamaindex/llm_gateway.js
```

### ❯ Sample Session

```
What topic do you want to ask about? artificial general intelligence
Temperature (0.0-2.0, default 0.7): 0.6
Max tokens (default 1000): 500
Query ALL providers or select one? (all/one): all
```
**Output:**

```
=== OpenAI GPT (LangChain) answered:
[temp: 0.6, max_tokens: 500, model: gpt-4o]
Artificial general intelligence (AGI) refers to...

=== Anthropic Claude (LangChain) answered:
[temp: 0.6, max_tokens: 500, model: claude-3-5-sonnet-20241022]
AGI is a type of AI that can perform...
```

* [Web API Mode]

**NOTE**: Only one **web server** can be active at a time, since all use **port 8000** by default.

### ❯ Example (Python)

```bash
python langchain/llm_gateway.py web
python llamaindex/llm_gateway.py web
```

### ❯ Example (JavaScript)

```bash
node langchain/llm_gateway.js web
node llamaindex/llm_gateway.js web
```

### ❯ Endpoints

| Method | Path         | Description                      |
| ------ | ------------ | -------------------------------- |
| GET    | `/`          | Service status and init messages |
| GET    | `/health`    | Health check                     |
| GET    | `/providers` | List initialized providers       |
| POST   | `/query`     | Query a single provider          |
| POST   | `/query-all` | Query all available providers    |

---

## API Examples (`curl`)

> Output is identical between Python and JS backends.

---

### ✅ Get Status

```bash
curl http://localhost:8000/
```

```json
{
  "framework": "LangChain",
  "available_providers": ["openai", "anthropic"],
  "total_available": 2,
  "status": "healthy"
}
```

---

### ✅ Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "framework": "LangChain",
  "providers_available": 2
}
```

---

### ✅ Provider Metadata

```bash
curl http://localhost:8000/providers
```

```json
{
  "framework": "LangChain",
  "providers": [
    {
      "name": "openai",
      "display_name": "OpenAI GPT",
      "model": "gpt-4o",
      "status": "✓ Initialized successfully"
    },
    {
      "name": "anthropic",
      "display_name": "Anthropic Claude",
      "model": "claude-3-5-sonnet-20241022",
      "status": "✓ Initialized successfully"
    }
  ]
}
```

---

### ✅ Query a Single Provider

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "What is machine learning?",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 300
      }'
```

```json
{
  "success": true,
  "provider": "openai",
  "model": "gpt-4o",
  "response": "Machine learning is a field of AI...",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 300
  },
  "prompt": "What is machine learning?"
}
```

---

### ✅ Query All Providers

```bash
curl -X POST http://localhost:8000/query-all \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "Explain the Turing Test",
        "temperature": 0.5,
        "max_tokens": 400
      }'
```

```json
{
  "success": true,
  "prompt": "Explain the Turing Test",
  "responses": {
    "openai": {
      "success": true,
      "response": "The Turing Test evaluates whether a machine can mimic human responses...",
      "model": "gpt-4o"
    },
    "anthropic": {
      "success": true,
      "response": "The Turing Test, proposed by Alan Turing...",
      "model": "claude-3-5-sonnet-20241022"
    }
  }
}
```
