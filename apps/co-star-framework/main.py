# co-star-framework.py
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Set up our Local LLM ---
# Make sure Ollama is running with `llama3:latest` pulled
# Command: ollama run llama3:latest
llm = ChatOllama(model="llama3:latest")

# --- 2. Define the technical content to be summarized ---
technical_log = """
feat(api): migrate user endpoint to new auth middleware

Refactored the primary user data endpoint (/api/v2/user) to use the new JWT-based authentication middleware (auth-v3). 
This change deprecates the legacy session-based validation logic. The new middleware enforces stricter scope validation 
(read:user, write:user) and integrates with the global rate-limiting service. 
Key changes include updating the request handler signature, removing the dependency on `express-session`, and adding 
`authV3.verifyToken()` to the route's middleware stack. All related integration tests were updated and are passing.
"""

def run_agent_prompt(prompt_template, content):
    """A simple function to run our agent with a given prompt template."""
    prompt = prompt_template.format_prompt(log_content=content).to_messages()
    
    # Simple chain: prompt -> llm -> output_parser
    chain = llm | StrOutputParser()
    
    print("--- AGENT PROMPT ---")
    print(prompt[0].content)
    print("\n--- AGENT RESPONSE ---")
    response = chain.invoke(prompt)
    print(response)
    print("-" * 20)

# --- 3. The BAD Prompt: Vague and Unstructured ---
bad_prompt_template = ChatPromptTemplate.from_template(
    "Summarize the following update log for my manager: {log_content}"
)

# Run the agent with the bad prompt
print("RUNNING BAD PROMPT...")
run_agent_prompt(bad_prompt_template, technical_log)

# --- 4. The BLUEPRINT Prompt: Precise and Structured ---
blueprint_prompt_template = ChatPromptTemplate.from_template(
    """
### CONTEXT ###
You are an expert engineering lead communicating with a non-technical Product Manager. 
You need to translate technical software updates into clear, impact-oriented business language. 
Avoid jargon. Focus on the "so what?". The following is a git commit log for a recent update.

Technical Log:
"{log_content}"

### AUDIENCE ###
The audience is a Product Manager who is not a software developer. They care about product stability, security, and future capabilities, not implementation details.

### STYLE & TONE ###
Your style should be clear, concise, and professional. The tone should be informative and confident, assuring the manager that the change is positive.

### OBJECTIVE & RESPONSE FORMAT ###
Your objective is to summarize the technical log for the Product Manager.
You MUST provide the response as a Markdown formatted list with the following three headers exactly:
- **What's New:** (A one-sentence, high-level summary of the change.)
- **Business Impact:** (Explain the benefits, such as improved security or performance, in 2-3 bullet points.)
- **Action Required:** (State if the manager needs to do anything. If not, state "None.")
    """
)

# Run the agent with the blueprint prompt
print("\nRUNNING BLUEPRINT PROMPT...")
run_agent_prompt(blueprint_prompt_template, technical_log)
