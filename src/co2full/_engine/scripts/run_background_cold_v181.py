import math

from subtasks.methods import COMPILE_V2


def content(row):
    return COMPILE_V2.format(code_a=row["function_a"]["pseudo_code"], code_b=row["function_b"]["pseudo_code"])


def gates(parent, structure, frozen):
    a = structure["score"][frozen["selected"]["low_source"]]
    b = structure["score"][frozen["selected"]["high_source"]]
    cause = structure.get("cause") or structure["arms"]["structural_yes_no_then_models"]["cause"]
    terminal = parent["terminal"].removesuffix("_repair")
    base = (
        parent["source"] == "model"
        and parent["prediction"] == 0
        and cause == "density_weak_compile_disabled"
        and a is not None
        and b is not None
        and math.isfinite(a)
        and math.isfinite(b)
        and a >= 0
        and b >= 0
    )
    return dict(
        fewshot=bool(base and terminal == "fewshot"),
        expanded=bool(base and terminal in ["fewshot", "type117", "flow121"]),
    )
