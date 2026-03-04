from pathlib import Path
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import Field
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Chemin absolu vers chroma_db (fonctionne quel que soit le repertoire d'execution)
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")

# --- Tool qui interroge ton ChromaDB existant ---
class ChromaSearchTool(BaseTool):
    name: str = "recherche_documents"
    description: str = "Recherche des informations dans les documents PDF indexés. Utile pour répondre à des questions factuelles."

    def _run(self, query: str) -> str:
        client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client_chroma.get_collection("documents")

        openai_client = OpenAI()
        embedding = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[embedding],
            n_results=3
        )

        chunks = results["documents"][0]
        return "\n\n---\n\n".join(chunks)

# --- Les agents ---
researcher = Agent(
    role="Researcher",
    goal="Trouver des informations précises dans les documents disponibles",
    backstory="Tu es expert en recherche documentaire. Tu utilises TOUJOURS l'outil de recherche avant de répondre.",
    tools=[ChromaSearchTool()],
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Rédiger une réponse claire basée sur la recherche",
    backstory="Tu transformes des extraits bruts en réponse lisible et synthétique.",
    verbose=True
)

# --- Change la question ici pour tester ---
QUESTION = "Quelle est la population de Besançon ?"

task_research = Task(
    description=f"Réponds à cette question en cherchant dans les documents : {QUESTION}",
    expected_output="Les extraits pertinents trouvés dans les documents.",
    agent=researcher
)

task_write = Task(
    description=f"Rédige une réponse finale claire à cette question : {QUESTION}",
    expected_output="Une réponse en 2-3 phrases, basée uniquement sur les documents.",
    agent=writer,
    context=[task_research]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    verbose=True
)

result = crew.kickoff()
print("\n--- RÉPONSE FINALE ---")
print(result)
