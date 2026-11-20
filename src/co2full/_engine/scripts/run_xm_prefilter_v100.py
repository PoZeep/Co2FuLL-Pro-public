from co2full.runtime import tool_path as _cf_tool_path

import json

import baseline_fewshot as base

from subtasks.methods import COMPILE_V2

REVIEW = """Review the previous few-shot source-origin judgment against the two complete decompiled functions below. The previous answer is fallible, not a label. Give a new final decision; either YES or NO may be revised.
1. Identify the concrete code premise carrying the previous explanation, and check whether the code actually supports it.
2. If the explanation rejects a match because of control-flow shape, expression/return form, inferred types/signatures, anonymous callees, field offsets or temporary names, check whether a specific compilation/decompilation transformation explains that difference while preserving the distinctive computation. State the correspondence supporting the explanation. Do not invent a helper's behavior or assume every difference is a compiler effect; different tested conditions, constants, input transformations or essential writes may matter.
3. Independently check the positive case: shared purpose, a common loop or one library call alone does not establish common source. Look for a distinctive correspondence of operations, data flow and side effects. Finding an error in the old reason does not by itself justify YES. Conversely, do not demand identical syntax, known anonymous callees or an instruction-level proof when the core computation aligns.
Resolve the balance of code evidence and output your own judgment. In at most 300 words, give the old-premise check, the compilation explanation (or substantive mismatch), and the source-specific correspondence (or generic similarity). Finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source> tag.
"""


def messages(row, arm):
    a, b = [row[s]["pseudo_code"] for s in ("function_a", "function_b")]
    if arm == "reason_reflection":
        prior = dict(
            previous_prediction="yes" if row["original"]["prediction"] else "no",
            previous_explanation=row["original"]["reason"],
        )
        content = (
            REVIEW
            + "\nPrevious answer:\n"
            + json.dumps(prior, ensure_ascii=False)
            + "\n\nCode A:\n"
            + a
            + "\n\nCode B:\n"
            + b
        )
    else:
        content = COMPILE_V2.format(code_a=a, code_b=b)
        if arm == "old_reason_compile":
            content = (
                "Auxiliary analysis may contain errors. Check it against the code; it is not a source-origin label. "
                "Do not infer same source merely because an earlier explanation was wrong.\n"
                + json.dumps(dict(previous_explanation=row["original"]["reason"]), ensure_ascii=False)
                + "\n\n"
                + content
            )
        else:
            assert arm == "compile_control"
    return [dict(role="system", content=base.SYSTEM_PROMPT), dict(role="user", content=content)]


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_cf_tool_path("model")), local_files_only=True)


def token_count(tok, mm):
    return len(tok.apply_chat_template(mm, tokenize=True, add_generation_prompt=True))
