# Direct LLM API access: Anthropic, OpenAI, Google, xAI

This chapter demonstrates how to access the major LLM providers— Anthropic, OpenAI, Google, and xAI—using their in-house APIs and minimal working scripts in both Python and JavaScript.

## Project Contents
Each subfolder reflects an LLM provider and contains:

- A `nursery.py` script (Python)
- A `nursery.js` script (JavaScript)
- The necessary `package.json` / `requirements.txt` for dependencies

The scripts generate and test completions using the nursery rhyme:  
`"Twinkle, Twinkle, Little"`, to verify LLM functionality and output quality.

**NOTE**: (An `.env`- file with the corresponding API key must also be generated)
---


## Environment Setup

For each provider (`anthropic`, `openai`, `google`, `xai`):

### ❯ Example (Python)

### Step 1. Navigate into the provider folder:
   ```bash
   cd anthropic  # or openai, google, xai
````

### Step 2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Step 3. Create an `.env` file with the correct API key:

   ```env
   ANTHROPIC_API_KEY="sk-ant-api03-xxxxxx"
   OPENAI_API_KEY="sk-proj-xxxx-xxxx-xxxx"
   GOOGLE_API_KEY="AIzaSyBAKuqxxxxxxxxxxxx"
   XAI_API_KEY="xai-UXxxxxxxx"
   ```

### Step 4. Run the script:

   ```bash
   python nursery.py
   ```

---

### > Example (JavaScript)

### Step 1. Navigate into the provider folder:

   ```bash
   cd anthropic  # or openai, google, xai
   ```

### Step 2. Install dependencies:

   ```bash
   npm install
   ```

### Step 3. Create an `.env` file with the correct API key (same format as above).

### Step 4. Run the script:

   ```bash
   node nursery.js
   ```

---

## Summary

Each script:

* Loads its API key from `.env`
* Sends the prompt `"Twinkle, Twinkle, Little"` to the model
* Outputs the model's prediction

Each script validates the correct API integration with each provider, outputs fidelity across models using the same prompt and requires the minimum setup requirements for Python or JavaScript.

This setup also gives a clear baseline for model behavior across providers before using higher-level frameworks like LangChain or LlamaIndex.
