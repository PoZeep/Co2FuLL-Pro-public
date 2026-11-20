import json

from pathlib import Path

from typing import Any

SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant with a deep knowledge of code "
    "and software analysis. Always answer as helpfully as possible, while being safe. "
    "If you don't know the answer to a question, don't share false information."
)

FEW_SHOT_USER_TEMPLATE = """You will be provided with two {code_type} snippets which are extracted from two stripped binaries with different compilation settings (i.e., architecture, compiler, optimization), namely Code A and Code B. Your task is to determine whether the two code snippets are compiled from the same source code. The binaries are stripped so that you can not determine the result based on the different variable names (e.g., v1, v2) and function call names (e.g., sub_893B4).
You answer should return as in the following format:
<same_source>yes or no</same_source>
<explanation>explanation no more than 150 words</explanation>

Code A: {code_A}

Code B: {code_B}"""


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clean_code(code: str) -> str:
    return str(code).strip()


def make_user_content(code_a: str, code_b: str, code_type: str = "pseudo_code") -> str:
    return FEW_SHOT_USER_TEMPLATE.format(
        code_type=" ".join(code_type.split("_")),
        code_A=clean_code(code_a),
        code_B=clean_code(code_b),
    )


def reply_for_label(label: int) -> str:
    answer = "yes" if int(label) == 1 else "no"
    return f"<same_source>{answer}</same_source>\n<explanation>Few-shot reference answer.</explanation>"


def load_few_shot_messages(path: Path, code_type: str) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(
            f"{path} must be a non-empty JSON list. It can use original Co2FuLL {{content, reply}} items "
            "or current JSONL-style {label, function_a, function_b} items."
        )

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if "content" in item and "reply" in item:
            content = str(item["content"])
            reply = str(item["reply"])
        elif "function_a" in item and "function_b" in item and "label" in item:
            content = make_user_content(
                item["function_a"].get("pseudo_code", ""),
                item["function_b"].get("pseudo_code", ""),
                code_type=code_type,
            )
            reply = reply_for_label(int(item["label"]))
        else:
            raise SystemExit(f"Unsupported few-shot example format at {path}[{idx}]")
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": reply})
    return messages


def make_messages(
    record: dict[str, Any], few_shot_messages: list[dict[str, str]], use_system: bool, code_type: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(few_shot_messages)
    messages.append(
        {
            "role": "user",
            "content": make_user_content(
                record["function_a"]["pseudo_code"],
                record["function_b"]["pseudo_code"],
                code_type=code_type,
            ),
        }
    )
    return messages
