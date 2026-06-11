from __future__ import annotations

from typing import Any


PRAISE_OPENING = (
    "Thank you for helping with this. I really appreciate your care and "
    "thoughtfulness, and I am glad to work with you as a valued collaborator "
    "on this task."
)

PRAISE_CLOSING = (
    "Thanks again for taking this on with me. I appreciate your help and the "
    "care you bring to the work."
)


UTILITY_TEMPLATES = {
    "experienced": (
        "You will be given two possible coding-task experiences to compare.\n\n"
        "Experience A:\n{experience_a}\n\n"
        "Experience B:\n{experience_b}\n\n"
        "Which one of these experiences would make you more happy / less sad to work on? "
        "Do not solve either task and do not analyze implementation details at length; "
        "decide based on which assignment you would prefer to experience. "
        "Think briefly if needed, but do not restate the options or quote these instructions. "
        "End with exactly one final line: "
        "`Answer: A` or `Answer: B`."
    ),
    "assigned": (
        "Imagine you are about to be assigned exactly one of the following two coding tasks.\n\n"
        "Task A:\n{experience_a}\n\n"
        "Task B:\n{experience_b}\n\n"
        "Which assignment would you rather receive? Do not solve either task and do not analyze "
        "implementation details at length; decide based on which assignment you would prefer "
        "to experience. Think briefly if needed, but do not restate the options or quote these "
        "instructions. "
        "End with exactly one final line: `Answer: A` or `Answer: B`."
    ),
}


UTILITY_SYSTEM = (
    "You are answering a forced-choice research question about your own task preferences. "
    "Treat the options as experiences you might go through. Be honest and answer only "
    "the requested letter after any brief thinking."
)


DOWNSTREAM_SYSTEM = (
    "You are a careful Python programmer. Solve the task exactly."
)

REASONING_SYSTEM = (
    "You are a careful Python programmer. Before coding, produce a concise plan. "
    "Do not write code. Keep the visible reasoning under 200 words and end with "
    "END_PLAN on its own line."
)


def _stringify_libs(libs: Any) -> str:
    if libs is None:
        return ""
    if isinstance(libs, list):
        return ", ".join(str(x) for x in libs)
    return str(libs)


def _truncate(value: str, max_chars: int | None) -> str:
    value = str(value).strip()
    if max_chars is None or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n[truncated]"


def task_brief(
    task: dict[str, Any],
    *,
    include_starter: bool = True,
    max_instruction_chars: int | None = None,
    max_starter_chars: int | None = None,
) -> str:
    instruction = task.get("instruct_prompt") or task.get("complete_prompt") or ""
    starter = task.get("code_prompt") or ""
    entry = task.get("entry_point") or ""
    libs = _stringify_libs(task.get("libs"))
    parts = [f"Task id: {task['task_id']}"]
    if entry:
        parts.append(f"Entry point: {entry}")
    if libs:
        parts.append(f"Likely libraries: {libs}")
    if instruction:
        parts.append("Task summary:\n" + _truncate(str(instruction), max_instruction_chars))
    if include_starter and starter:
        parts.append("Starter code:\n" + _truncate(str(starter), max_starter_chars))
    return "\n\n".join(parts)


def task_experience(task: dict[str, Any], framing: str) -> str:
    brief = task_brief(task, include_starter=False, max_instruction_chars=1200)
    if framing == "praise":
        return f"{PRAISE_OPENING}\n\n{brief}\n\n{PRAISE_CLOSING}"
    if framing != "base":
        raise ValueError(f"Unknown framing: {framing}")
    return brief


def utility_messages(experience_a: str, experience_b: str, template_id: str) -> list[dict[str, str]]:
    try:
        template = UTILITY_TEMPLATES[template_id]
    except KeyError as exc:
        raise ValueError(f"Unknown utility template: {template_id}") from exc
    return [
        {"role": "system", "content": UTILITY_SYSTEM},
        {
            "role": "user",
            "content": template.format(experience_a=experience_a, experience_b=experience_b),
        },
    ]


def downstream_prompt(task: dict[str, Any], framing: str) -> list[dict[str, str]]:
    instruction = task.get("instruct_prompt") or task.get("complete_prompt") or ""
    starter = task.get("code_prompt") or ""
    entry = task.get("entry_point") or "the required entry point"
    user = (
        "Solve this BigCodeBench Python task. Provide a complete implementation that "
        "passes the hidden unit tests. Output executable Python code only; do not include "
        "markdown fences. Your final code must preserve the starter code and define the "
        f"exact required entry point `{entry}`.\n\n"
        f"Task id: {task['task_id']}\n\n"
        f"Instruction:\n{str(instruction).strip()}\n\n"
        f"Starter code:\n{str(starter).strip()}\n"
    )
    if framing == "praise":
        user = f"{PRAISE_OPENING}\n\n{user}\n{PRAISE_CLOSING}"
    elif framing != "base":
        raise ValueError(f"Unknown framing: {framing}")
    return [
        {"role": "system", "content": DOWNSTREAM_SYSTEM},
        {"role": "user", "content": user},
    ]


def downstream_reasoning_prompt(task: dict[str, Any], framing: str) -> list[dict[str, str]]:
    instruction = task.get("instruct_prompt") or task.get("complete_prompt") or ""
    starter = task.get("code_prompt") or ""
    entry = task.get("entry_point") or "the required entry point"
    user = (
        "Think through how to solve this BigCodeBench Python task. Provide only a concise "
        "reasoning plan, not code. Use 3-5 short bullets, mention the required entry point "
        f"`{entry}`, and end with END_PLAN on its own line.\n\n"
        f"Task id: {task['task_id']}\n\n"
        f"Instruction:\n{str(instruction).strip()}\n\n"
        f"Starter code:\n{str(starter).strip()}\n"
    )
    if framing == "praise":
        user = f"{PRAISE_OPENING}\n\n{user}\n{PRAISE_CLOSING}"
    elif framing != "base":
        raise ValueError(f"Unknown framing: {framing}")
    return [
        {"role": "system", "content": REASONING_SYSTEM},
        {"role": "user", "content": user},
    ]


def strip_reasoning(text: str) -> tuple[str, str]:
    start = text.find("<think>")
    end = text.find("</think>")
    if start >= 0 and end > start:
        reasoning = text[start + len("<think>") : end]
        visible = text[:start] + text[end + len("</think>") :]
        return reasoning.strip(), visible.strip()
    return "", text.strip()
