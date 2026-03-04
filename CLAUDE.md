# RAG from scratch — projet d'apprentissage

## Stack
- Python 3.11
- openai, chromadb, pypdf, tiktoken

## Structure
- 01_ingest.py : chunking + embedding + stockage ChromaDB
- 02_search.py : recherche sémantique sur une question
- 03_generate.py : génération LLM avec les chunks récupérés

## Variables d'env requises
- OPENAI_API_KEY

## Contraintes
- Pas de framework (pas LangChain, pas LlamaIndex)
- Scripts indépendants, pas de classes inutiles
- Commentaires en français
