import ast

import re

from collections import Counter

from subtasks.readonly_evidence_v72 import read_elf


def mask(code):
    return re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda m: "".join("\n" if c == "\n" else " " for c in m[0]),
        code,
        flags=re.S,
    )


def signature(code):
    header = code.split("{", 1)[0]
    m = re.search(r"\((.*)\)\s*$", header, re.S)
    if not m:
        return None
    prefix = header[: m.start()]
    ret = re.sub(r"\b(?:__cdecl|__fastcall|__stdcall)\b", "", prefix)
    ret = re.sub(r"\b\w+\s*$", "", ret).strip()
    ret = re.sub(r"\s+", " ", ret)
    params = []
    for part in m[1].split(","):
        part = part.strip()
        p = re.search(r"\b([A-Za-z_]\w*)\s*$", part.strip())
        if not p:
            return None
        type_name = re.sub(r"\s+", " ", part[: p.start()].strip())
        type_name = re.sub(r"\s*\*\s*", "*", type_name)
        params.append(dict(name=p[1], type=type_name))
    return ret, params


def polynomial(expression, params, width):
    """Normalize supported unsigned expressions modulo 2**width."""
    roles = {p["name"]: (i, p["type"]) for i, p in enumerate(params)}
    expression = re.sub(r"(?<=\d)[uUlL]+\b", "", expression.strip())
    node = ast.parse(expression, mode="eval").body
    modulus = 1 << width
    stats = Counter()

    def normalize(value):
        return {k: v % modulus for k, v in value.items() if v % modulus}

    def add(a, b, sign=1):
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + sign * v
        return normalize(result)

    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return {(): n.value % modulus}
        if isinstance(n, ast.Name):
            i, kind = roles[n.id]
            assert kind == "unsigned int" and width == 32
            stats["atoms"] += 1
            return {(f"arg{i}",): 1}
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            i, kind = roles[n.value.id]
            assert kind == "unsigned int*" and width == 32
            sub = n.slice
            if hasattr(ast, "Index") and isinstance(sub, ast.Index):
                sub = sub.value
            assert isinstance(sub, ast.Constant) and isinstance(sub.value, int) and sub.value >= 0
            stats["atoms"] += 1
            return {(f"arg{i}[{sub.value}]",): 1}
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            value = walk(n.operand)
            return normalize({k: (-1 if isinstance(n.op, ast.USub) else 1) * v for k, v in value.items()})
        assert isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult))
        stats["operations"] += 1
        a, b = walk(n.left), walk(n.right)
        if isinstance(n.op, ast.Add):
            return add(a, b)
        if isinstance(n.op, ast.Sub):
            return add(a, b, -1)
        result = Counter()
        for ka, va in a.items():
            for kb, vb in b.items():
                result[tuple(sorted(ka + kb))] += va * vb
                assert len(result) <= 128
        return normalize(result)

    result = walk(node)
    assert len(result) <= 128 and stats["operations"] >= 2
    assert len({x for term in result for x in term}) >= 2
    return [dict(factors=list(k), coefficient=v) for k, v in sorted(result.items())], dict(stats)


def return_facts(a, b):
    parsed = [signature(x) for x in [a, b]]
    if any(x is None for x in parsed):
        return []
    (ra, pa), (rb, pb) = parsed
    if ra not in ["int", "unsigned int"] or rb not in ["int", "unsigned int"]:
        return []
    if [x["type"] for x in pa] != [x["type"] for x in pb]:
        return []
    collected = []
    for code, params in [(a, pa), (b, pb)]:
        out = []
        for m in list(re.finditer(r"\breturn\s+([^;{}]+);", mask(code)))[:8]:
            expr = code[m.start(1) : m.end(1)].strip()
            try:
                form, stats = polynomial(expr, params, 32)
            except (AssertionError, SyntaxError, ValueError, KeyError, RecursionError, OverflowError):
                continue
            out.append(
                dict(expression=expr, line=code.count("\n", 0, m.start()) + 1, normal_form=form, stats=stats)
            )
        collected.append(out)
    result = []
    for x in collected[0]:
        for y in collected[1]:
            if x["normal_form"] == y["normal_form"] and re.sub(r"\s+", "", x["expression"]) != re.sub(
                r"\s+", "", y["expression"]
            ):
                result.append(
                    dict(
                        kind="unsigned_return_polynomial_identity",
                        width=32,
                        a=x,
                        b=y,
                        input_roles=[
                            dict(role=i, a=p["name"], b=q["name"], type=p["type"])
                            for i, (p, q) in enumerate(zip(pa, pb))
                        ],
                        scope="Only these pure return expressions under positional input-role correspondence and unsigned32 arithmetic modulo2^32. Constant-index unsigned32 memory reads are treated as corresponding values. Calls, other paths, writes, pointer identity, volatility and whole-function origin are NOT proved. The model must check that the expressions occur on corresponding paths.",
                    )
                )
    return result[:4]


def matching_brace(code, start):
    depth = 0
    for p in range(start, len(code)):
        if code[p] == "{":
            depth += 1
        elif code[p] == "}":
            depth -= 1
            if depth == 0:
                return p
    return None


def table_accesses(code):
    parsed = signature(code)
    if not parsed:
        return []
    params = parsed[1]
    byte_params = {
        x["name"]: i for i, x in enumerate(params) if x["type"] in ["unsigned __int8*", "unsigned char*"]
    }
    source = mask(code)
    loads = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*\*(\w+)\s*;", source):
        if m[2] in byte_params:
            assignments = re.findall(r"\b" + re.escape(m[1]) + r"\s*(?:=(?!=)|\+=|-=|\*=|/=|\+\+|--)", source)
            if len(assignments) == 1:
                loads[m[1]] = dict(
                    role=byte_params[m[2]],
                    parameter=m[2],
                    load_line=source.count("\n", 0, m.start()) + 1,
                    position=m.end(),
                )
    result = []
    for m in re.finditer(r"\b(word_([0-9a-fA-F]+))\s*\[([^\]]+)\]", source):
        idx = re.sub(r"\s+", "", m[3])
        for variable, load in loads.items():
            if m.start() <= load["position"]:
                continue
            k = re.fullmatch(re.escape(variable) + r"(?:([+-])(0[xX][0-9a-fA-F]+|\d+)[uUlL]*)?", idx)
            if not k:
                continue
            shift = (int(k[2], 0) if k[2] else 0) * (-1 if k[1] == "-" else 1)
            domain = set(range(256))
            guards = []
            pattern = (
                r"\bif\s*\(\s*"
                + re.escape(variable)
                + r"\s*(<=|>=|<|>|==|!=)\s*(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\s*\)\s*\{"
            )
            for g in re.finditer(pattern, source):
                start = g.end() - 1
                end = matching_brace(source, start)
                if end is None or not (start < m.start() < end):
                    continue
                value = int(g[2], 0)
                operator = g[1]
                keep = {
                    "<": lambda x: x < value,
                    ">": lambda x: x > value,
                    "<=": lambda x: x <= value,
                    ">=": lambda x: x >= value,
                    "==": lambda x: x == value,
                    "!=": lambda x: x != value,
                }[operator]
                domain = {x for x in domain if keep(x)}
                guards.append(
                    dict(
                        expression=code[g.start() : g.end() - 1].strip(),
                        line=code.count("\n", 0, g.start()) + 1,
                    )
                )
            if not guards or not domain or min(domain) + shift < 0:
                continue
            result.append(
                dict(
                    identifier=m[1],
                    base_address=int(m[2], 16),
                    element_bytes=2,
                    index_expression=code[m.start(3) : m.end(3)],
                    input_role=load["role"],
                    input_parameter=load["parameter"],
                    loaded_variable=variable,
                    affine_shift=shift,
                    domain=sorted(domain),
                    guards=guards,
                    reference_line=code.count("\n", 0, m.start()) + 1,
                )
            )
    return result[:4]


def read_table(access, elf, domain):
    cells = []
    for value in domain:
        address = access["base_address"] + 2 * (value + access["affine_shift"])
        section = next(
            (
                s
                for s in elf["sections"]
                if s["address"] <= address and address + 2 <= s["address"] + s["size"]
            ),
            None,
        )
        if section is None:
            return None
        offset = section["offset"] + address - section["address"]
        raw = elf["raw"][offset : offset + 2]
        cells.append(
            dict(
                input=value,
                address=address,
                file_offset=offset,
                bytes_hex=raw.hex(),
                value=int.from_bytes(raw, "little" if elf["endian"] == "<" else "big"),
                section=section["name"],
            )
        )
    return cells


def indexed_data_facts(a, b, elf_a, elf_b):
    facts = []
    for x in table_accesses(a):
        for y in table_accesses(b):
            if x["input_role"] != y["input_role"]:
                continue
            domain = sorted(set(x["domain"]) & set(y["domain"]))
            if not domain:
                continue
            aa, bb = read_table(x, elf_a, domain), read_table(y, elf_b, domain)
            if aa is None or bb is None:
                continue
            differences = [
                dict(input=t["input"], a=t["value"], b=u["value"])
                for t, u in zip(aa, bb)
                if t["value"] != u["value"]
            ]
            facts.append(
                dict(
                    kind="indexed_readonly_word_values",
                    a=dict(access=x, cells=aa),
                    b=dict(access=y, cells=bb),
                    domain=domain,
                    different_entries=len(differences),
                    first_differences=differences[:12],
                    scope="Actual file bytes at the decompiled16-bit table accesses, aligned by the same unsigned-byte input role and intersection of recognized enclosing positive-if guards. Other guards may further restrict execution. The decompiler-implied16-bit load, byte-input type and positional role mapping are assumptions explicitly shown. These values do not establish complete object boundaries, all-path equivalence, or source origin. The model must check whether differing entries affect observed outputs on reachable paths. No symbols or function identities were read.",
                )
            )
    return facts[:4]


def selftest():
    params = [dict(name="a", type="unsigned int*"), dict(name="n", type="unsigned int")]
    x, _ = polynomial("8*a[1]*n+a[2]", params, 32)
    y, _ = polynomial("a[2]+a[1]*8*n", params, 32)
    z, _ = polynomial("a[2]+a[1]*4*n", params, 32)
    assert x == y and x != z
    p, _ = polynomial("(a[1]+a[2])*n", params, 32)
    q, _ = polynomial("a[1]*n+a[2]*n", params, 32)
    assert p == q
    for expr in ["helper(a[1])+a[2]*n", "a[1]/n+a[2]*n", "8.0*a[1]*n+a[2]", "a+n*a[1]"]:
        try:
            polynomial(expr, params, 32)
        except (AssertionError, KeyError, SyntaxError):
            pass
        else:
            raise AssertionError(("unsupported expression accepted", expr))
    code = "int f(int ignored, unsigned int *out, unsigned __int8 *src) { unsigned int v; v = *src; if (v > 0x9F) { *out = word_1000[v - 160]; return 1; } return 0; }"
    access = table_accesses(code)
    assert (
        len(access) == 1
        and access[0]["domain"] == list(range(160, 256))
        and access[0]["affine_shift"] == -160
    )
    raw = b"".join(i.to_bytes(2, "little") for i in range(96))
    elf = dict(raw=raw, endian="<", sections=[dict(address=0x1000, offset=0, size=len(raw), name=".rodata")])
    shifted = dict(access[0], base_address=0x1000 - 320, affine_shift=0)
    aa = read_table(access[0], elf, access[0]["domain"])
    bb = read_table(shifted, elf, access[0]["domain"])
    assert aa == bb and [x["value"] for x in aa] == list(range(96))
    assert read_table(access[0], elf, [256]) is None
    return dict(
        unsigned_ring_identity=True,
        unequal_expression_rejected=True,
        unsupported_syntax_rejected=True,
        guarded_byte_domain=True,
        base_shift_alignment=True,
        section_bounds=True,
    )


def model_facts(facts, swapped=False):
    result = []
    for f in facts:
        a, b = (f["b"], f["a"]) if swapped else (f["a"], f["b"])
        if f["kind"] == "unsigned_return_polynomial_identity":
            roles = [dict(x, a=x["b"], b=x["a"]) if swapped else dict(x) for x in f["input_roles"]]
            result.append(
                dict(
                    kind=f["kind"],
                    width=f["width"],
                    expression_a=a["expression"],
                    expression_b=b["expression"],
                    line_a=a["line"],
                    line_b=b["line"],
                    input_roles=roles,
                    common_polynomial=a["normal_form"],
                    scope=f["scope"],
                )
            )
        else:
            assert f["kind"] == "indexed_readonly_word_values"
            keep = [
                "identifier",
                "index_expression",
                "input_role",
                "input_parameter",
                "loaded_variable",
                "affine_shift",
                "guards",
                "reference_line",
            ]
            diff = [dict(x, a=x["b"], b=x["a"]) if swapped else dict(x) for x in f["first_differences"]]
            result.append(
                dict(
                    kind=f["kind"],
                    access_a={k: a["access"][k] for k in keep},
                    access_b={k: b["access"][k] for k in keep},
                    input_values=f["domain"],
                    values_a=[x["value"] for x in a["cells"]],
                    values_b=[x["value"] for x in b["cells"]],
                    different_entries=f["different_entries"],
                    first_differences=diff,
                    scope=f["scope"],
                )
            )
    return result


def content(row, variant):
    import json
    from subtasks.methods import COMPILE_V2

    assert variant in ["plain", "data", "swapped"]
    a, b = row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"]
    if variant == "plain":
        return COMPILE_V2.format(code_a=a, code_b=b)
    swapped = variant == "swapped"
    if swapped:
        a, b = b, a
    instruction = (
        "The following narrow facts were mechanically computed from the displayed code or actual readonly binary bytes. "
        "They are evidence about specific operations, NOT source-origin labels. Check the explicitly stated assumptions and locate each fact in both complete functions. "
        "A matched return expression proves nothing about other paths or calls; equal table entries do not prove whole-function identity. "
        "If table entries differ, check whether those indexed entries reach a return or output write on a reachable input, rather than dismissing the difference merely as a changed address. "
        "If an integer expression identity applies, do not call harmless rearrangement of that expression a behavioral difference. "
        "Combine these facts with the complete code to make your own final judgment.\n"
    )
    return (
        instruction
        + json.dumps(model_facts(row["facts"], swapped), ensure_ascii=False)
        + "\n\n"
        + COMPILE_V2.format(code_a=a, code_b=b)
    )
