# ollama_utils

A simple utility package for initializing Ollama LLM and embedding models for use in LangChain-based Python applications.

## Features

- Centralizes Ollama model initialization for reuse across multiple apps in your monorepo.
- Easily configurable: choose any LLM or embedding model supported by your Ollama server.
- Handles connection errors gracefully.

## Usage

Import and use the utility in your app:

```python
from ollama_utils import get_ollama_models

# Use default models
llm, embeddings = get_ollama_models()

# Or specify custom models and base URL
llm, embeddings = get_ollama_models(
    llm_model="llama2:latest",
    embedding_model="nomic-embed-text",
    base_url="http://localhost:11434"
)
```

- `llm` is a `ChatOllama` instance (for chat/completion).
- `embeddings` is an `OllamaEmbeddings` instance (for vector DBs).

If the Ollama server is not running or the model is unavailable, the function will print an error and exit.

## Running the Tests

This package includes unit tests using `pytest` and `unittest.mock`.

To run the tests from the root of your monorepo:

```sh
poetry run pytest packages/ollama_utils/
```

The tests mock Ollama classes, so you do not need a running Ollama server to test the utility logic.

---

For more information about Ollama, see: https://ollama.com/
