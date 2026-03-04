"""
03_generate.py — Génération RAG : recherche les chunks pertinents puis
envoie le tout à gpt-4o-mini pour obtenir une réponse contextualisée.
"""

import os
import sys

import chromadb
from openai import OpenAI

# --- Configuration ---
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"
N_RESULTS = 3

# --- Récupération de la question ---
if len(sys.argv) < 2:
    print("Usage : python 03_generate.py \"Votre question ici\"")
    sys.exit(1)

question = sys.argv[1]

# --- Recherche des chunks pertinents (même logique que 02_search.py) ---
client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client_chroma.get_collection(name=COLLECTION_NAME)

# Embedding de la question via OpenAI
client_openai = OpenAI()
embedding_resp = client_openai.embeddings.create(
    input=question,
    model="text-embedding-3-small",
)
query_embedding = embedding_resp.data[0].embedding

# Recherche sémantique dans ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=N_RESULTS,
)

chunks = results["documents"][0]

# --- Construction du prompt ---
contexte = "\n---\n".join(chunks)

prompt = f"""Réponds à cette question en te basant uniquement sur ces extraits.
Si les extraits ne contiennent pas la réponse, dis-le clairement.

Extraits :
{contexte}

Question : {question}"""

# --- Appel à gpt-4o-mini ---
response = client_openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu es un assistant qui répond en français en se basant sur le contexte fourni."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.3,
)

# --- Affichage de la réponse ---
print("\n=== Réponse ===\n")
print(response.choices[0].message.content)
