from subtasks.methods import COMPILE_V2

from subtasks.representation_review_v115 import content as original_review


def content(row, arm):
    variant = arm.rsplit("_", 1)[0]
    if variant == "repeat_with_reason":
        return original_review(row, "control_1")
    a = row["function_a"]["pseudo_code"]
    b = row["function_b"]["pseudo_code"]
    if variant == "blind_reverse":
        a, b = b, a
    else:
        assert variant == "blind_forward"
    return COMPILE_V2.format(code_a=a, code_b=b)
