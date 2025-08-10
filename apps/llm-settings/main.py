# agent.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the LLM
# This connects to the Ollama service running on your local machine
llm = ChatOllama(model="llama3:latest")

# 2. Create a Prompt Template
# This defines the input structure for the model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides clear and concise answers."),
    ("user", "{input}")
])

# 3. Create a simple chain
# This links the prompt and the model together
chain = prompt | llm | StrOutputParser()

# Example of how to run the chain
# print(chain.invoke({"input": "What is the capital of France?"}))

# A creative prompt
# question = "Describe, in one short, poetic phrase, the feeling of the first sip of 'Nebula Roast' coffee on a quiet morning."

# print(f"PROMPT: {question}\n" + "="*30)

# # Low temperature -> predictable and safe
# llm_low_temp = ChatOllama(model="llama3", temperature=0.0)
# response_low = llm_low_temp.invoke(question)
# print(f"Low Temp Tagline: {response_low.content}")

# print("-"*30)

# # High temperature -> creative and surprising
# llm_high_temp = ChatOllama(model="llama3", temperature=2.0)
# response_high = llm_high_temp.invoke(question)
# print(f"High Temp Tagline: {response_high.content}")

# Using top_p to get a creative but coherent answer
# We are creating a 'nucleus' of the most likely tokens that add up to 90% probability
# top_p_llm = ChatOllama(model="llama3:latest", top_p=0.9)

# response = top_p_llm.invoke("The future of artificial intelligence is ")
# print(f"Top-p (0.9) Response: {response.content}")

# Using bind_options to apply parameters at runtime
# num_predict=-1 means no limit, a positive integer sets the token limit.
# For Ollama, pass num_predict directly to the constructor.
# limited_llm = ChatOllama(model="llama3:latest", num_predict=50)

# response = limited_llm.invoke("Summarize the concept of Retrieval-Augmented Generation (RAG) in one paragraph.")
# print(f"Limited Response (50 tokens): {response.content}")
# Expected output will be cut off abruptly after about 50 tokens.

# Without penalties, the model might get stuck in a loop
# repetitive_llm = ChatOllama(model="llama3:latest", temperature=0.5)
# response = repetitive_llm.invoke("Write a long paragraph about the benefits of open-source software.")
# print(f"Without Penalty:\n{response.content}\n")

# With frequency and presence penalties to encourage new ideas
# less_repetitive_llm = ChatOllama(
#     model="llama3:latest",
#     temperature=0.5,
#     # Ollama uses 'repeat_penalty' which combines aspects of both
#     # A value > 1.0 penalizes repetition.
#     repeat_penalty=1.5 
# )
# response_penalized = less_repetitive_llm.invoke("Write a long paragraph about the benefits of open-source software.")
# print(f"With Penalty:\n{response_penalized.content}")
# Expected output should be more varied and less repetitive.

# We want the agent to stop generating as soon as it thinks of writing "User:"
stop_llm = ChatOllama(
    model="llama3:latest",
    # stop=["User:", "\n\nHuman:"] # A list of stop sequences
)

# A prompt that might tempt the model to continue the conversation
chat_prompt = "You are a helpful AI assistant. Complete your answer and nothing more.\n\nHuman: What is LangChain?\nAssistant:"
response = stop_llm.invoke(chat_prompt)
print(response.content)
# The output should stop cleanly without adding a fake "User:" turn.
