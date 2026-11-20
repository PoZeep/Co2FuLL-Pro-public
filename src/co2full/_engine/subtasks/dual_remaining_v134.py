from subtasks.methods import COMPILE_V2

TAIL = "\nEach candidate contains renderings of the same physical function, not extra source functions. Use Ghidra only to check representation uncertainty in IDA. Its anonymous identifiers are local to that rendering; do not equate them across candidates or invent behavior for opaque calls."


def content(row, method):
    kind, stage, repeat = method.split("_")
    assert kind in ["control", "context"] and stage in ["raw", "confirm"]

    def code(side):
        original = row[side]["pseudo_code"]
        return (
            original
            if kind == "control"
            else "Original IDA view:\n"
            + original
            + "\nGhidra view of this SAME physical function (local identifiers anonymized):\n"
            + row["alternate"][side]
        )

    a, b = code("function_a"), code("function_b")
    if stage == "confirm":
        a, b = b, a
    return COMPILE_V2.format(code_a=a, code_b=b) + (TAIL if kind == "context" else "")
