from crewai import Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv()

# 1. L'agent
agent = Agent(
    role="Analyste",
    goal="Répondre à des questions de manière concise",
    backstory="Tu es un assistant factuel et direct.",
    verbose=True  # affiche le raisonnement interne de l'agent
)

# 2. La tâche
task = Task(
    description="Explique en 3 phrases ce qu'est un RAG.",
    expected_output="Un texte de 3 phrases maximum.",
    agent=agent
)

# 3. La crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

# 4. On lance
result = crew.kickoff()
print("\n--- RÉSULTAT ---")
print(result)
