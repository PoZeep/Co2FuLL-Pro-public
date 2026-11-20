from subtasks.methods import COMPILE_V2

NOTE = (
    "\nThe tree is an observed IDA instruction ctree, not ground-truth source or a normalized algorithm. "
    "Node identifiers are local to each tree. Indentation records syntactic nesting, not semantic equivalence. "
    "Different loop/branch shapes alone do not prove different source. Compare actual distinctive computations and effects. "
    "Do not invent unknown helper behavior or infer cross-function identity from local names.\n"
)


def content(row, method):
    kind, stage, repeat = method.split("_")
    assert kind in ["control", "tree", "context"] and stage in ["raw", "confirm"]
    if stage == "confirm":
        return COMPILE_V2.format(
            code_a=row["function_b"]["pseudo_code"], code_b=row["function_a"]["pseudo_code"]
        )

    def code(side):
        original = row[side]["pseudo_code"]
        tree = row["trees"][side]
        if kind == "control":
            return original
        if kind == "tree":
            header = original.split("{", 1)[0].rstrip()
            return "Original function declaration:\n" + header + "\nComplete IDA instruction ctree:\n" + tree
        return (
            "Original pseudocode:\n"
            + original
            + "\nComplete IDA instruction ctree of this SAME function:\n"
            + tree
        )

    return COMPILE_V2.format(code_a=code("function_a"), code_b=code("function_b")) + (
        NOTE if kind != "control" else ""
    )


def render(nodes):
    by = {n["node_id"]: n for n in nodes}
    assert len(by) == len(nodes) and nodes
    children = {k: [] for k in by}
    roots = []
    for n in nodes:
        p = n.get("parent_id")
        if p is None:
            roots.append(n["node_id"])
        else:
            assert p in by
            children[p].append(n["node_id"])
    assert len(roots) == 1
    out = []
    seen = set()

    def visit(key, depth):
        assert key not in seen
        seen.add(key)
        n = by[key]
        kind = n["kind"].upper()
        if kind == "FOR":
            body = (
                "FOR (" + n.get("init", "") + "; " + n.get("condition", "") + "; " + n.get("step", "") + ")"
            )
        elif "condition" in n:
            body = kind + " (" + n["condition"] + ")"
        elif n.get("text"):
            body = kind + ": " + n["text"]
        else:
            body = kind
        out.append("  " * depth + "[" + str(key) + "] " + body)
        for child in children[key]:
            visit(child, depth + 1)

    visit(roots[0], 0)
    assert seen == set(by)
    return "\n".join(out)
