from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# Agent 1 : cherche des infos
researcher = Agent(
    role="Researcher",
    goal="Trouver des informations précises sur un sujet",
    backstory="Tu es expert en recherche et synthèse d'information.",
    verbose=True
)

# Agent 2 : rédige à partir du travail du premier
writer = Agent(
    role="Writer",
    goal="Rédiger un texte clair à partir d'une recherche",
    backstory="Tu transformes des notes brutes en texte lisible.",
    verbose=True
)

# Tâche 1 : assignée au researcher
task_research = Task(
    description="Explique comment fonctionne ChromaDB en tant que base vectorielle.",
    expected_output="Une liste de 5 points clés sur ChromaDB.",
    agent=researcher
)

# Tâche 2 : assignée au writer, qui reçoit l'output de la tâche 1
task_write = Task(
    description="Rédige un paragraphe de présentation de ChromaDB pour un PM non-technique.",
    expected_output="Un paragraphe de 5 phrases maximum, sans jargon.",
    agent=writer,
    context=[task_research]  # 👈 c'est ici que le Writer reçoit le travail du Researcher
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    verbose=True
)

result = crew.kickoff()
print("\n--- RÉSULTAT FINAL ---")
print(result)
