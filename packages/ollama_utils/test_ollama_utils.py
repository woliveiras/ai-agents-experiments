import pytest
from unittest.mock import patch
from . import get_ollama_models

@patch("ollama_utils.ChatOllama")
@patch("ollama_utils.OllamaEmbeddings")
def test_get_ollama_models_success(mock_embeddings, mock_llm):
    llm_instance = mock_llm.return_value
    embeddings_instance = mock_embeddings.return_value
    llm, embeddings = get_ollama_models("custom-llm", "custom-embed", "http://fake-url")
    assert llm is llm_instance
    assert embeddings is embeddings_instance

@patch("ollama_utils.ChatOllama", side_effect=Exception("fail"))
def test_get_ollama_models_fail(mock_llm):
    import sys
    with pytest.raises(SystemExit):
        get_ollama_models()
