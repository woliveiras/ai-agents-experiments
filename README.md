# ai-agents-experiments

A monorepo for experimenting with AI agents using Ollama, LangChain, and ChromaDB. This project demonstrates how to build, run, and extend AI-powered applications leveraging modern LLMs and vector databases, with a focus on practical, reproducible workflows for developers and students.

## Features

- **Ollama**: Run open-source LLMs locally in Docker.
- **LangChain**: Build retrieval-augmented generation (RAG) and agent pipelines.
- **ChromaDB**: Store and query vector embeddings for semantic search.
- **Modular Apps**: Each app is isolated in its own folder under `apps/`.
- **Reproducible Environment**: Managed with Poetry and Docker Compose.

## Requirements

- [Docker](https://docs.docker.com/desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Poetry](https://python-poetry.org/docs/)
- Python 3.13+

## Getting Started

### 1. Start the Infrastructure

```sh
docker-compose up -d
```

This will start the Ollama and ChromaDB services in the background.

### 2. Install Ollama Models

```sh
docker exec -it ollama_service ollama pull llama3:latest
docker exec -it ollama_service ollama pull nomic-embed-text
```

### 3. Install Python Dependencies

```sh
poetry install
```

### 4. Set Up Environment Variables

Create a `.env` file in `apps/github-qa-agent/` with your GitHub token:

```
GITHUB_ACCESS_TOKEN=your_github_token_here
```

### 5. Install Internal Utilities

After installing dependencies, install the internal utility package in editable mode:

```sh
poetry run pip install -e packages/
```

### 6. Run the Apps

```sh
poetry run python apps/github_qa_agent/main.py
poetry run python apps/tree_of_thoughts_agent/main.py
poetry run python apps/langgraph_lats_agent/main.py
poetry run python apps/vet_crew/main.py
```

### 7. Running Tests

To run all tests (including for internal packages):

```sh
poetry run pytest
```

## Project Structure

```
.
├── apps/
│   └── github-qa-agent/    # Example: QA agent for GitHub repos
├── data/                   # Vector DB and source docs
├── docker-compose.yml      # Infra services
├── pyproject.toml          # Python/Poetry config
├── README.md
└── ...
```

## Example: GitHub QA Agent

This app indexes a GitHub repository, stores embeddings in ChromaDB, and lets you ask questions about the codebase using a local LLM.

- Loads code from a public repo
- Splits and embeds documents
- Stores/retrieves vectors via ChromaDB (Docker)
- Answers questions using Ollama LLM

Reference: [How to Chat with Your GitHub Repository: A Guide to Local RAG with Ollama and LangChain](https://woliveiras.github.io/posts/how-to-chat-with-github-repository-a-guide-to-local-rag-with-ollama-and-langchain/).

## Extending

- Add new apps under `apps/`
- Use any LLM supported by Ollama
- Swap vector DBs or add new chains with LangChain

## License

MIT. See [LICENSE](./LICENSE).

---

For students and beginners: This monorepo is a great starting point for learning about LLMs, vector search, and building AI-powered tools with modern Python workflows.