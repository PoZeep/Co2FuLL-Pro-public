import re

from subtasks.representation_review_v115 import TRIGGER, content as reason_content

from subtasks.methods import COMPILE_V2

FLOW = re.compile(
    r"\b(?:control.flow|loops?|branches|branching|branch|iterations?|recurs\w*|unroll\w*|vector\w*|SIMD|conditions?)\b",
    re.I,
)


def eligible(route, reason):
    old = route["arms"]["structural_yes_no_then_models"]
    return (
        old["cause"] == "density_weak_compile_disabled"
        and route["fewshot"] == 0
        and not TRIGGER.search(reason)
        and bool(FLOW.search(reason))
    )


def content(row, method):
    kind, stage, repeat = method.split("_")
    assert kind in ["reason", "cold"] and stage in ["raw", "confirm"] and repeat in ["1", "2"]
    if kind == "reason" and stage == "raw":
        return reason_content(row, "control_1")
    a, b = row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"]
    if (kind, stage) in [("reason", "confirm"), ("cold", "raw")]:
        a, b = b, a
    return COMPILE_V2.format(code_a=a, code_b=b)
