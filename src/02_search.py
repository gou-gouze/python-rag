"""
02_search.py — Recherche sémantique dans ChromaDB.

Prend une question, l'embed via OpenAI, et retourne
les 3 chunks les plus proches avec leur score de similarité.
"""

import os
import sys

import chromadb
from openai import OpenAI

# --- Configuration ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
MODEL_EMBEDDING = "text-embedding-3-small"
TOP_K = 3

# --- Vérification clé API ---
if not os.getenv("OPENAI_API_KEY"):
    print("Erreur : OPENAI_API_KEY non définie.")
    sys.exit(1)

client_openai = OpenAI()
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)

# --- Récupération de la collection ---
try:
    collection = client_chroma.get_collection(name=COLLECTION_NAME)
except Exception:
    print(f"Erreur : collection '{COLLECTION_NAME}' introuvable dans {CHROMA_DIR}/.")
    print("Lance d'abord 01_ingest.py pour indexer des documents.")
    sys.exit(1)

print(f"Collection '{COLLECTION_NAME}' : {collection.count()} chunks indexés.\n")

# --- Question ---
question = "Quels sont les principaux monuments historiques d'Istanbul ?"

print(f"Question : {question}\n")

# --- Embedding de la question ---
response = client_openai.embeddings.create(
    input=question,
    model=MODEL_EMBEDDING,
)
query_embedding = response.data[0].embedding

# --- Recherche des chunks les plus proches ---
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=["documents", "metadatas", "distances"],
)

# --- Affichage des résultats ---
documents = results["documents"][0]
distances = results["distances"][0]
metadatas = results["metadatas"][0]

if not documents:
    print("Aucun résultat trouvé.")
    sys.exit(0)

for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas), 1):
    # ChromaDB retourne des distances L2 par défaut ;
    # plus la distance est petite, plus le chunk est pertinent.
    print(f"--- Résultat {i} (distance : {dist:.4f}) ---")
    if meta:
        print(f"    Source : {meta.get('source', '?')}  |  Chunk #{meta.get('chunk_index', '?')}")
    print(f"    {doc[:300]}{'…' if len(doc) > 300 else ''}")
    print()
