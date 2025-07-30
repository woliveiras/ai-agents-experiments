# tree_of_thoughts_agent.py
from ollama_utils import get_ollama_models

def generate_thoughts(goal: str, count: int) -> list[str]:
    """Generates multiple distinct 'thoughts' or solutions for a given goal."""
    print(f"\n--- 1. GENERATING {count} THOUGHTS ---")
    prompt = f"You are a creative writer. Generate {count} completely different and compelling opening lines for a story about: '{goal}'. Each opening line should be on a new line, starting with '1. ', '2. ', etc."

    llm, _ = get_ollama_models()
    response = llm.invoke(prompt)

    # Simple parsing: split by newline and remove numbering
    lines = response.content.strip().split('\n')
    thoughts = [line.split('. ', 1)[1] for line in lines if '. ' in line]
    print("Generated options:")
    for t in thoughts:
        print(f"- {t}")
    return thoughts

def evaluate_thoughts(goal: str, thoughts: list[str]) -> str:
    """Evaluates a list of thoughts and selects the best one."""
    print("\n--- 2. EVALUATING THOUGHTS ---")
    # Construct a prompt for the LLM to act as a judge
    evaluation_prompt = f"""
    You are a literary critic. Your task is to evaluate the following opening lines for a story about '{goal}'.
    Which one is the most engaging and creates the most intrigue? Explain your reasoning and then state the best option at the end in the format 'BEST OPTION: [The full text of the best opening line]'.

    Here are the options:
    """
    for i, thought in enumerate(thoughts, 1):
        evaluation_prompt += f"\n{i}. {thought}"

    llm, _ = get_ollama_models()
    response = llm.invoke(evaluation_prompt)
    evaluation_text = response.content
    print(f"EVALUATION: {evaluation_text}")

    # Extract the best option from the judge's response
    best_option_line = [line for line in evaluation_text.split('\n') if line.startswith("BEST OPTION:")]
    if best_option_line:
        best_option = best_option_line[0].replace("BEST OPTION:", "").strip()
        if not best_option:
            # Try to get the next non-empty line after 'BEST OPTION:'
            lines = evaluation_text.split('\n')
            idx = lines.index(best_option_line[0])
            for next_line in lines[idx+1:]:
                if next_line.strip():
                    best_option = next_line.strip()
                    break
        return best_option
    return "No best option determined."

if __name__ == "__main__":
    user_goal = "A detective hunting a rogue AI in a rain-drenched, neon-lit city."

    print(f"--- GOAL: {user_goal} ---")

    # 1. Generate multiple initial thoughts (branches of the tree)
    generated_lines = generate_thoughts(user_goal, 3)

    # 2. Evaluate the thoughts and select the most promising branch
    best_line = evaluate_thoughts(user_goal, generated_lines)

    print("\n--- 3. FINAL SELECTED PATH ---")
    print(f"The most promising opening line is: '{best_line}'")