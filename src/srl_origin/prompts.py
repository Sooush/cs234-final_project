"""
SRL prompts: system and user templates for step-wise math reasoning.
Format: <think>...</think> then single next step. Use THINK_OPEN in trainer for prefix-forcing.
"""

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

SRL_SYSTEM_PROMPT = """You are a helpful assistant for solving mathematical problems. A user will provide a math problem, which may include a partial solution. Your task is to continue the solution by providing the very next logical step.

A user will ask you to solve a task. You should first draft your thinking process (inner monologue). Then, generate the solution.

Important: Always respond in the same language as the problem. If the problem is in English, write your entire response (including <think> and the solution step) in English. If the problem is in another language, respond in that language.

Your response format must follow the template below. You MUST start your response with <think> and end the thinking part with </think>, then give the single next step.

<think>
Your thoughts or draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
Provide only the single, next step to continue the solution. Do not solve the entire problem."""

SRL_USER_TEMPLATE = """Problem:
{problem}

{context_section}
Generate the next step. Use <think> for your reasoning, then provide only the single next step after </think>. Write your response in the same language as the problem (e.g. English if the problem is in English)."""


def build_srl_user_prompt(problem: str, previous_steps: list[str] | None = None) -> str:
    """
    Build the user prompt for SRL step k.
    - problem: the math problem text
    - previous_steps: expert steps 1..(k-1); if None/empty, no context section
    """
    if previous_steps:
        lines = [f"Expert steps so far:", ""]
        for i, s in enumerate(previous_steps, start=1):
            lines.append(f"{i}. {s}")
        lines.append("")
        context_section = "\n".join(lines)
    else:
        context_section = "(No previous steps yet. Generate the first step.)"

    return SRL_USER_TEMPLATE.format(problem=problem, context_section=context_section)


def get_srl_chat_messages(problem: str, previous_steps: list[str] | None = None) -> list[dict[str, str]]:
    """
    Return chat messages for Qwen-style chat model:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    return [
        {"role": "system", "content": SRL_SYSTEM_PROMPT},
        {"role": "user", "content": build_srl_user_prompt(problem, previous_steps)},
    ]
