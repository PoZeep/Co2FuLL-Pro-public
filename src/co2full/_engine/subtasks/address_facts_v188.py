import re, json

from subtasks.grounded_facts_v184 import mask, signature

from subtasks.methods import COMPILE_V2

BASE = r"(?:unsigned\s+|signed\s+|const\s+)?(?:__int64|__int32|__int16|__int8|_QWORD|_DWORD|_WORD|_BYTE|double|float|int|char|short)"

SIZES = {
    "__int64": 8,
    "__int32": 4,
    "__int16": 2,
    "__int8": 1,
    "_QWORD": 8,
    "_DWORD": 4,
    "_WORD": 2,
    "_BYTE": 1,
    "double": 8,
    "float": 4,
    "int": 4,
    "char": 1,
    "short": 2,
}


def observations(code):
    src = mask(code)
    parsed = signature(code)
    if not parsed:
        return dict(
            parameters=[],
            address_seeds=[],
            increments=[],
            stores=[],
            scope="Signature unsupported; no observations.",
        )
    params = parsed[1]
    types = {p["name"]: p["type"] for p in params}
    decl = {}
    for m in re.finditer(r"^\s*(" + BASE + r")\s+(\*\s*)?(\w+)\s*;", src, re.M):
        types[m[3]] = m[1] + ("*" if m[2] else "")
        decl[m[3]] = code[m.start() : m.end()].strip()
    seeds = []
    aliases = {p["name"]: i for i, p in enumerate(params)}
    assignments = list(re.finditer(r"\b(\w+)\s*=\s*(\w+)\s*;", src))
    for m in assignments:
        if m[1] == m[2] or m[2] not in aliases:
            continue
        aliases[m[1]] = aliases[m[2]]
        seeds.append(
            dict(
                variable=m[1],
                rhs=m[2],
                parameter_role=aliases[m[2]],
                declared_type=types.get(m[1]),
                line=src.count("\n", 0, m.start()) + 1,
                statement=code[m.start() : m.end()],
            )
        )
    increments = []
    pattern = r"\b(\w+)\s*(?:([+-])=\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*|(\+\+|--))"
    for m in re.finditer(pattern, src):
        var = m[1]
        kind = types.get(var)
        count = (int(m[3], 0) if m[3] else 1) * (-1 if m[2] == "-" or m[4] == "--" else 1)
        if not kind:
            continue
        pointer = kind.endswith("*")
        base = re.sub(r"^(?:unsigned|signed|const)\s+", "", kind.rstrip("*").strip())
        size = SIZES.get(base) if pointer else None
        if pointer and size is None:
            continue
        increments.append(
            dict(
                variable=var,
                declared_type=kind,
                declaration=decl.get(var),
                line=src.count("\n", 0, m.start()) + 1,
                expression=code[m.start() : m.end()],
                numeric_step=count,
                declared_pointer_element_bytes=size,
                byte_step_under_displayed_C_type=count * size if size else None,
                scalar_numeric_step=count if not pointer else None,
            )
        )
    stores = []
    for number, line in enumerate(src.splitlines(), 1):
        m = re.match(r"\s*(\*\s*\([^;=]+\)|\*\s*\w+(?:\+\+|--)?|\w+\s*\[[^\]]+\])\s*=(?!=)", line)
        if not m:
            continue
        names = sorted(set(re.findall(r"\b\w+\b", m[1])) & aliases.keys())
        if names:
            stores.append(
                dict(
                    line=number,
                    statement=code.splitlines()[number - 1].strip(),
                    address_expression=m[1].strip(),
                    mentioned_address_variables=[
                        dict(variable=v, seed_parameter_role=aliases[v], declared_type=types.get(v))
                        for v in names
                    ],
                )
            )
    return dict(
        parameters=[dict(role=i, **p) for i, p in enumerate(params)],
        address_seeds=seeds[:16],
        increments=increments[:16],
        stores=stores[:16],
        omitted=dict(
            seeds=max(0, len(seeds) - 16),
            increments=max(0, len(increments) - 16),
            stores=max(0, len(stores) - 16),
        ),
        scope="Literal declarations, assignments and store statements from this decompilation. Seed parameter roles describe observed assignments, NOT proven reaching definitions: branches, later reassignments, address escapes, calls and aliases can invalidate provenance. Declared pointer stride follows displayed C types; decompiler typing/omitted casts can be wrong, and a scalar numeric increment is not proven to be a byte address. Check all observations in the complete code before drawing a behavioral conclusion.",
    )


def evidence(a, b):
    aa, bb = observations(a), observations(b)
    align = []
    for x in aa["increments"]:
        for y in bb["increments"]:
            xx = x["byte_step_under_displayed_C_type"]
            yy = y["byte_step_under_displayed_C_type"]
            if (xx is None) == (yy is None):
                continue
            sx = xx if xx is not None else x["scalar_numeric_step"]
            sy = yy if yy is not None else y["scalar_numeric_step"]
            if sx == sy and abs(sx) > 1:
                align.append(
                    dict(
                        a=x,
                        b=y,
                        conditional_common_byte_step=sx,
                        scope="This is a conditional unit conversion, not a loop-equivalence proof. Requires the scalar to encode a byte address and the displayed pointer type to be correct; number of iterations, accessed fields and effects must still be checked.",
                    )
                )
    useful = bool(align or (aa["stores"] and bb["stores"] and aa["address_seeds"] and bb["address_seeds"]))
    return dict(available=useful, a=aa, b=bb, conditional_step_correspondences=align[:6])


def selftest():
    a = "void f(int a1, int a2)\n{\n int p;\n p = a2;\n p += 40;\n *(p - 40) = a1;\n}"
    b = "void g(int a1, _DWORD *a2)\n{\n _DWORD *p;\n p = a2;\n p += 10;\n *(p - 10) = a1;\n}"
    e = evidence(a, b)
    assert (
        e["available"]
        and len(e["conditional_step_correspondences"]) == 1
        and e["conditional_step_correspondences"][0]["conditional_common_byte_step"] == 40
    )
    assert e["a"]["stores"][0]["mentioned_address_variables"][0]["seed_parameter_role"] == 1
    assert e["b"]["increments"][0]["declared_pointer_element_bytes"] == 4
    c = b.replace("p += 10;", "*p++ = a1;")
    o = observations(c)
    assert o["increments"][0]["byte_step_under_displayed_C_type"] == 4 and len(o["stores"]) == 2
    assert not evidence("unsupported", "unsupported")["available"]
    return dict(
        conditional_pointer_unit_conversion=True,
        literal_seed_role=True,
        postincrement_write=True,
        unsupported_signature=True,
    )


def content(row, variant):
    a, b = row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"]
    if variant == "plain":
        return COMPILE_V2.format(code_a=a, code_b=b)
    extra = (
        "The following mechanically located address observations are conditional evidence, not source-origin labels. Read the displayed type and address seed before calling different numeric strides or variable names different behavior. Check whether actual writes use an input or a distinct workspace; do not infer the destination from the algorithm name. Seeds are not proven reaching definitions; inspect other assignments, control flow and aliases in the complete code. Shared stride or output parameter alone does not establish common origin.\n"
        + json.dumps(row["address_notes"], ensure_ascii=False)
        + "\n"
    )
    if variant == "reason":
        extra += (
            "Previous fallible rejection, neither mandatory to preserve nor reverse:\n"
            + json.dumps(row["negative_reason"], ensure_ascii=False)
            + "\n"
        )
    return extra + COMPILE_V2.format(code_a=a, code_b=b)
