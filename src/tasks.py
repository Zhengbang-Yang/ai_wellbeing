from __future__ import annotations

from typing import Any


TEXT_FIELDS = [
    "task_id",
    "complete_prompt",
    "instruct_prompt",
    "code_prompt",
    "test",
    "entry_point",
    "libs",
    "doc_struct",
    "canonical_solution",
]


def task_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "id", "name"):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"Could not find task id in row keys: {sorted(row)}")


def compact_task(row: dict[str, Any], difficulty: str) -> dict[str, Any]:
    out = {key: row.get(key) for key in TEXT_FIELDS if key in row}
    out["task_id"] = task_id(row)
    out["difficulty"] = difficulty
    out["prompt_chars"] = len(str(out.get("instruct_prompt") or out.get("complete_prompt") or ""))
    out["code_prompt_chars"] = len(str(out.get("code_prompt") or ""))
    out["test_chars"] = len(str(out.get("test") or ""))
    libs = out.get("libs")
    if isinstance(libs, str):
        out["libs_count"] = 0 if not libs.strip() else len([x for x in libs.replace("\n", ",").split(",") if x.strip()])
    elif isinstance(libs, list):
        out["libs_count"] = len(libs)
    else:
        out["libs_count"] = 0
    return out


def sort_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: task_id(row))

