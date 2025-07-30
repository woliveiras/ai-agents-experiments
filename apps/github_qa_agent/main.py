import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import chromadb
from ollama_utils import get_ollama_models

# --- 1. Load Environment Variables ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

if not GITHUB_TOKEN:
    print("❌ GITHUB_ACCESS_TOKEN not found. Please create a .env file in 'apps/github-qa-agent/'")
    sys.exit(1)


# --- 2. Define Constants ---
REPO_URL = "https://github.com/woliveiras/reader-agent"
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma-db-data" / "github-repo-agent"

def main():
    """
    Main function to set up and run the GitHub QA agent.
    """
    print("🚀 Starting the AI agent for GitHub repository analysis...")

    # --- 3. Initialize Models ---
    llm, embeddings = get_ollama_models()

    # --- 4. Load or Create the Vector Database ---
    # Use HTTP client for ChromaDB running in Docker
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection_name = "github-repo-agent"
    try:
        # Try to connect to existing collection
        print(f"✅ Connecting to ChromaDB at http://localhost:8000...")
        vector_store = Chroma(
            embedding_function=embeddings,
            client=chroma_client,
            collection_name=collection_name
        )
        # Optionally, check if collection is empty and create if needed
        if len(vector_store.get()['ids']) == 0:
            raise ValueError("ChromaDB collection is empty, will create new one.")
    except Exception as e:
        print(f"⏳ Creating new vector database for {REPO_URL} via HTTP...")
        print(f"   (This will be saved in the ChromaDB Docker volume)")
        # Load the repository files
        print("   Loading files from the GitHub repository...")
        loader = GithubFileLoader(
            repo="woliveiras/reader-agent",
            branch="main",
            access_token=GITHUB_TOKEN,
            github_api_url="https://api.github.com",
            file_filter=lambda file_path: file_path.endswith((".py", ".md", ".toml", ".lock")),
        )
        docs = loader.load()
        print(f"   Loaded {len(docs)} documents from the repository.")
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        print(f"   Documents split into {len(splits)} chunks.")
        # Create and persist the vector store via HTTP
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name
        )
        print("✅ Vector database created and persisted in ChromaDB Docker volume.")

    # --- 5. Create the RAG Chain ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert software developer. Answer the user's questions based ONLY on the provided context, "
            "which is the source code of a GitHub repository. "
            "If the answer is not in the context, state that you cannot answer based on the available information.\n\n"
            "CONTEXT:\n{context}"
        )),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # --- 6. Start Interactive Chat Loop ---
    print("\n✅ Agent is ready. Ask me anything about the repository code!")
    while True:
        try:
            question = input("➡️ Your question (or 'exit' to quit): ")
            if question.strip().lower() == 'exit':
                print("👋 Exiting agent...")
                break
            if not question.strip():
                continue

            print("...🤔 Querying the agent...")
            result = retrieval_chain.invoke({"input": question})

            print("\n--- Agent Response ---")
            print(result["answer"])
            print("----------------------\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting agent...")
            break

if __name__ == "__main__":
    main()