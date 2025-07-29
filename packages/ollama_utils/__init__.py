from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
import sys

LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

def get_ollama_models(llm_model=LLM_MODEL, embedding_model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL):
    """
    Initialize and return Ollama LLM and Embeddings objects.
    Raises SystemExit if Ollama is not available.
    """
    try:
        llm = ChatOllama(model=llm_model, base_url=base_url)
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        return llm, embeddings
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("   Is the Ollama container running? (docker-compose up -d)")
        sys.exit(1)
