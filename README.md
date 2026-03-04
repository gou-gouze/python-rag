# RAG from scratch en Python

Un pipeline **Retrieval-Augmented Generation (RAG)** minimaliste, construit from scratch en Python sans framework (pas de LangChain, pas de LlamaIndex).

Le but : comprendre chaque brique du RAG en les codant soi-même.

## Comment ca marche ?

Le RAG fonctionne en 3 etapes :

```
PDF → Chunks → Embeddings → ChromaDB    (ingestion)
Question → Embedding → Recherche ChromaDB    (recherche)
Question + Chunks pertinents → LLM → Reponse    (generation)
```

1. **Ingestion** : les PDFs sont decoupes en morceaux (chunks) de 512 tokens, transformes en vecteurs (embeddings) via OpenAI, puis stockes dans une base vectorielle ChromaDB.
2. **Recherche** : une question est transformee en vecteur, puis comparee aux chunks pour trouver les plus proches (similarite cosinus).
3. **Generation** : les chunks pertinents sont envoyes comme contexte a gpt-4o-mini, qui genere une reponse basee uniquement sur ces extraits.

## Stack technique

| Composant | Role |
|---|---|
| Python 3.11+ | Langage |
| OpenAI API | Embeddings (`text-embedding-3-small`) + Generation (`gpt-4o-mini`) |
| ChromaDB | Base de donnees vectorielle (stockage local) |
| pypdf | Extraction de texte depuis les PDFs |
| tiktoken | Comptage de tokens pour le decoupage |
| python-dotenv | Chargement des variables d'environnement |

## Structure du projet

```
python-rag/
├── documents/          # Deposer vos PDFs ici
├── src/
│   ├── 01_ingest.py    # Chunking + embedding + stockage ChromaDB
│   ├── 02_search.py    # Recherche semantique sur une question
│   └── 03_generate.py  # Generation LLM avec les chunks recuperes
├── chroma_db/          # Base vectorielle (generee automatiquement)
├── requirements.txt
├── .env                # Cle API OpenAI (a creer, non versionne)
└── README.md
```

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/gou-gouze/python-rag.git
cd python-rag
```

### 2. Creer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

### 4. Configurer la cle API OpenAI

Creer un fichier `.env` a la racine du projet :

```
OPENAI_API_KEY=sk-your-key-here
```

Vous pouvez obtenir une cle sur [platform.openai.com](https://platform.openai.com/api-keys).

### 5. Ajouter des documents

Deposer un ou plusieurs fichiers **PDF** dans le dossier `documents/`. Le script d'ingestion prendra automatiquement tous les `.pdf` du dossier.

## Utilisation

Les scripts se lancent dans l'ordre :

### Etape 1 : Ingestion des documents

```bash
python src/01_ingest.py
```

Extrait le texte des PDFs, les decoupe en chunks de 512 tokens (avec 64 tokens de chevauchement), genere les embeddings et les stocke dans ChromaDB.

### Etape 2 : Recherche semantique

```bash
python src/02_search.py
```

Pose une question et affiche les chunks les plus pertinents avec leur score de similarite. La question peut etre passee en argument :

```bash
python src/02_search.py "Quelle est la population de Besancon ?"
```

### Etape 3 : Generation d'une reponse

```bash
python src/03_generate.py
```

Pose une question, recupere les chunks pertinents, puis genere une reponse via gpt-4o-mini. Comme pour la recherche, la question peut etre passee en argument :

```bash
python src/03_generate.py "Qui a invente l'iPhone ?"
```

## Configuration

Les parametres sont modifiables directement dans chaque script :

| Parametre | Fichier | Valeur par defaut | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `01_ingest.py` | 512 | Taille des chunks en tokens |
| `CHUNK_OVERLAP` | `01_ingest.py` | 64 | Chevauchement entre chunks en tokens |
| `EMBEDDING_MODEL` | tous | `text-embedding-3-small` | Modele d'embedding OpenAI |
| `TOP_K` | `02_search.py` | 10 | Nombre de resultats retournes |
| `N_RESULTS` | `03_generate.py` | 3 | Nombre de chunks envoyes au LLM |

## Licence

Projet d'apprentissage. Libre d'utilisation.
