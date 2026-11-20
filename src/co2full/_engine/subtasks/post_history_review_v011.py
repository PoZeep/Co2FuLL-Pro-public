import copy

import json

import re


def negative_gate(reason):
    text = reason.lower()
    common = re.search(
        r"(?:same|identical|similar)\s+(?:original\s+|core\s+|source\s+|mathematical\s+)*(?:operation|function|algorithm|computation|logic|structure)",
        text,
    )
    explained = re.search(
        r"(?:plausibl\w*|likely|due to|result\w* from|suggest\w*|because of)[^.\n]{0,100}(?:compil\w*|decompil\w*|architecture|optimization|calling convention)",
        text,
    )
    same_source = re.search(r"(?:are|be|from|of)\s+(?:the\s+)?same\s+(?:original\s+)?source", text)
    addresses = re.search(
        r"(?:different|differences in)[^.\n]{0,90}(?:addresses|function names|memory addresses)", text
    )
    return bool(common and (explained or same_source or addresses))


INSTRUCTION = """Determine whether the two complete decompiled functions originate from the same source function compiled differently. You are reviewing earlier model judgments, not voting among them. Earlier explanations can misread code or invent an optimization.
Use the code to write three SHORT items:
ALIGNMENT: Give the specific computation that corresponds, using logical inputs, outputs, update formulas and effect order. Generic purpose or common library calls are insufficient. Do not use the function's name as proof.
DISPUTE: Check the decisive disagreement in the history against exact code expressions. Track operands and the use of their results. Account for byte versus element offsets, 32/64-bit ABI, packed floating values, inlining, vectorization, and constant specialization only when the surrounding computation supports it. Different anonymous addresses, temporary types or loop shapes alone are not a source difference. Conversely, a changed arithmetic formula, comparison direction, meaningful constant, required output or persistent state update needs an actual correspondence; merely saying "optimization" does not explain it. Do not assume unknown helper behavior or invent removed operations.
RESOLUTION: State whether a concrete source-level incompatibility remains after alignment. A strong distinctive correspondence with only supported representation changes warrants YES. A shared template with a different required computation or effect warrants NO. Lack of a formal equivalence proof is not itself a contradiction. If an old explanation attributes all remaining differences to compilation, do not repeat NO without identifying a real remaining incompatibility in the code.
Use at most 220 words in total. Finish with exactly <same_source>yes</same_source> or <same_source>no</same_source>.
"""


def content(row, reverse=False):
    row = copy.deepcopy(row)
    if reverse:
        row["function_a"], row["function_b"] = row["function_b"], row["function_a"]
        for h in row["history"]:
            h["orientation"] = "reversed" if h["orientation"] == "same" else "same"
    history = [
        dict(
            judgment="YES" if h["prediction"] else "NO",
            explanation=h["reason"],
            orientation="Earlier A is current B; earlier B is current A."
            if h["orientation"] == "reversed"
            else "Same A/B order.",
        )
        for h in row["history"]
    ]
    return (
        INSTRUCTION
        + "\nFallible judgment history:\n"
        + json.dumps(history, ensure_ascii=False)
        + "\n\nCode A:\n"
        + row["function_a"]["pseudo_code"]
        + "\n\nCode B:\n"
        + row["function_b"]["pseudo_code"]
    )
