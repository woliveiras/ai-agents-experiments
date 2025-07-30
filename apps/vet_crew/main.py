# vet_crew_main.py
from crewai import Agent, Task, Crew, Process
from langchain_ollama import ChatOllama
from crewai.tools import BaseTool
from ddgs import DDGS

# Tool for web search related to veterinary information
class SearchTools(BaseTool):
    name: str = "Web Search Tool"
    description: str = "A tool to search the web for veterinary information. Use it to find potential causes for symptoms."

    def _run(self, query: str) -> str:
        print(f"--- TOOL: Searching for '{query}' ---")
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        return str(results) if results else "No results found."

# --- Instantiate the tool ---
search_tool = SearchTools()

# Initialize the local LLM with Ollama provider
ollama_model = ChatOllama(model="qwen3:latest")

# Define the veterinary agents
triage_vet = Agent(
    role='Triage Veterinarian',
    goal="Analyze the initial pet health query, determine the cat's life stage (kitten, adult, or senior), and delegate to the appropriate specialist.",
    backstory="You are an experienced vet who is the first point of contact at a busy clinic. Your strength is quickly assessing a situation and getting the case to the right expert.",
    verbose=True,
    allow_delegation=True,
    llm=ollama_model
)

kitten_vet = Agent(
    role='Pediatric Feline Veterinarian',
    goal='Diagnose and provide initial advice for health issues in kittens (cats under 1 year old).',
    backstory='You are a world-renowned specialist in kitten health, known for your ability to spot subtle signs of developmental or infectious diseases in young cats.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_model
)

adult_senior_vet = Agent(
    role='Adult and Geriatric Feline Veterinarian',
    goal='Diagnose and provide initial advice for health issues in adult and senior cats (1 year and older).',
    backstory='You have decades of experience treating adult and senior cats, with deep knowledge of age-related diseases like kidney failure, hyperthyroidism, and dental disease.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_model
)

# Define the diagnostic tasks for the crew
triage_task = Task(
    description="Analyze the user's query: '{query}'. Your first job is to determine the cat's life stage from the query. Based on the age, you MUST delegate the diagnostic task to either the 'Pediatric Feline Veterinarian' or the 'Adult and Geriatric Feline Veterinarian'. Do not try to diagnose yourself.",
    expected_output="A delegation action to the correct specialist with all the necessary context from the query.",
    agent=triage_vet
)

diagnostic_task = Task(
    description="A pet owner is concerned about their cat. The Triage Vet has passed this case to you. Analyze the full context of the query and provide a differential diagnosis. What are the most likely causes of the symptoms described? Use your search tool if needed. Conclude with clear advice on whether this is an emergency and what the owner's next steps should be.",
    expected_output="A detailed report with 2-3 likely diagnoses, an explanation for each, and clear, actionable advice for the pet owner.",
)

# Assemble and run the crew
crew = Crew(
    agents=[triage_vet, kitten_vet, adult_senior_vet],
    tasks=[triage_task, diagnostic_task],
    process=Process.hierarchical,
    manager_llm=ollama_model
)

def main():
    print("--- Starting the Veterinary Crew ---")
    user_query = "My 9-month-old cat, Whiskers, has been very lethargic and hasn't eaten anything for a day. What could be wrong?"
    result = crew.kickoff(inputs={'query': user_query})
    
    print("\n--- Crew Finished ---")
    print("\nFinal Diagnostic Report:")
    print(result)

if __name__ == "__main__":
    main()