"""
Ingestion de PDFs → chunks → embeddings → ChromaDB.

Pour chaque PDF dans /documents :
  1. Extraire le texte avec pypdf
  2. Découper en chunks de 512 tokens avec 64 tokens de chevauchement
  3. Embedder chaque chunk avec text-embedding-3-small
  4. Sauvegarder dans une collection ChromaDB
"""

import os
import sys
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

# --- Configuration ---
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 64     # tokens
ENCODING_NAME = "cl100k_base"  # encodeur tiktoken pour les modèles OpenAI récents


def extraire_texte_pdf(chemin_pdf: Path) -> str:
    """Extrait tout le texte d'un fichier PDF."""
    reader = PdfReader(chemin_pdf)
    texte = ""
    for page in reader.pages:
        contenu = page.extract_text()
        if contenu:
            texte += contenu + "\n"
    return texte


def decouper_en_chunks(texte: str, taille: int = CHUNK_SIZE, chevauchement: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en chunks de `taille` tokens avec `chevauchement` tokens de recouvrement."""
    enc = tiktoken.get_encoding(ENCODING_NAME)
    tokens = enc.encode(texte)

    chunks = []
    debut = 0
    while debut < len(tokens):
        fin = debut + taille
        chunk_tokens = tokens[debut:fin]
        chunks.append(enc.decode(chunk_tokens))
        debut += taille - chevauchement

    return chunks


def embedder_chunks(client: OpenAI, chunks: list[str]) -> list[list[float]]:
    """Appelle l'API OpenAI pour obtenir les embeddings d'une liste de chunks."""
    # L'API accepte une liste de textes en une seule requête
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    return [item.embedding for item in response.data]


def main():
    # Vérifier que le dossier documents existe et contient des PDFs
    if not DOCUMENTS_DIR.exists():
        print(f"Erreur : le dossier {DOCUMENTS_DIR} n'existe pas.")
        sys.exit(1)

    fichiers_pdf = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not fichiers_pdf:
        print(f"Aucun PDF trouvé dans {DOCUMENTS_DIR}")
        sys.exit(1)

    print(f"📂 {len(fichiers_pdf)} PDF(s) trouvé(s) dans {DOCUMENTS_DIR}")

    # Initialiser le client OpenAI et ChromaDB
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Recréer la collection à chaque ingestion (idempotent)
    chroma_client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in chroma_client.list_collections()
    ] else None
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf_path in fichiers_pdf:
        print(f"\n📄 Traitement de {pdf_path.name}...")

        # 1. Extraire le texte
        texte = extraire_texte_pdf(pdf_path)
        if not texte.strip():
            print(f"  ⚠️  Aucun texte extrait de {pdf_path.name}, ignoré.")
            continue

        # 2. Découper en chunks
        chunks = decouper_en_chunks(texte)
        print(f"  → {len(chunks)} chunk(s) créé(s)")

        # 3. Embedder les chunks
        embeddings = embedder_chunks(client, chunks)

        # 4. Sauvegarder dans ChromaDB
        ids = [f"{pdf_path.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    print(f"\n✅ Ingestion terminée : {total_chunks} chunks indexés dans ChromaDB ({CHROMA_DIR})")


if __name__ == "__main__":
    main()
