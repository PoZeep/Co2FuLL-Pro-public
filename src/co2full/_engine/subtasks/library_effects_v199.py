import re, json, copy

from subtasks.methods import COMPILE_V2

from subtasks.grounded_facts_v184 import mask

KNOWN = set(
    "abs labs llabs fabs fabsf fabsl sqrt sqrtf sqrtl cbrt cbrtf cbrtl pow powf powl exp expf expl exp2 exp2f log logf logl log2 log10 log1p expm1 sin sinf sinl cos cosf cosl tan tanf tanl asin acos atan atan2 sinh cosh tanh erf erfc tgamma lgamma floor ceil trunc round nearbyint rint fmod remainder fmax fmin fma copysign ldexp frexp modf hypot strlen strcmp strncmp strcasecmp strncasecmp strcoll strxfrm strcpy strncpy strcat strncat strchr strrchr strstr memcmp memcpy memmove memset memchr malloc calloc realloc free read write fread fwrite fgets fputs puts printf fprintf sprintf snprintf scanf sscanf fopen fclose fseek ftell qsort bsearch".split()
)


def calls(code):
    clean = mask(code)
    body = clean.find("{")
    found = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", clean[body + 1 :]):
        start = m.start() + body + 1
        name = m[1]
        canonical = name.removeprefix("j_").removeprefix("__builtin_")
        if canonical not in KNOWN:
            continue
        pos = m.end() + body + 1
        depth = 1
        while pos < len(clean) and depth:
            if clean[pos] == "(":
                depth += 1
            elif clean[pos] == ")":
                depth -= 1
            pos += 1
        if not depth:
            found.append(
                dict(
                    operation=canonical,
                    original_name=name,
                    start=start,
                    end=pos,
                    call=code[start:pos],
                    line=code.count("\n", 0, start) + 1,
                )
            )
    return found


def facts(row):
    return {s: calls(row[s]["pseudo_code"]) for s in ["function_a", "function_b"]}


def eligible(row):
    f = facts(row)
    a, b = [{x["operation"] for x in f[s]} for s in ["function_a", "function_b"]]
    return a != b and bool(a or b)


def content(row, variant, reverse=False):
    if reverse:
        row = copy.deepcopy(row)
        row["function_a"], row["function_b"] = row["function_b"], row["function_a"]
        row["previous_orientation"] = (
            "Previous A is current B; previous B is current A."
            if row["previous_orientation"] == "Same A/B order."
            else "Same A/B order."
        )
    a, b = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
    plain = COMPILE_V2.format(code_a=a, code_b=b)
    if variant == "plain":
        return plain
    assert variant in ["facts", "reason"]
    ff = facts(row)
    packet = {
        s: [dict(operation=x["operation"], line=x["line"], exact_call=x["call"]) for x in ff[k]]
        for s, k in [("A", "function_a"), ("B", "function_b")]
    }
    notice = (
        "An automatic scan extracted the following COMPLETE call expressions for known operations already visible in the supplied pseudocode. This is a lexical aid, not a dynamic symbol lookup, known source identity, dataflow proof or instruction execution. Check the surrounding full code for guards, operands and use of the result.\n"
        + json.dumps(packet, ensure_ascii=False)
        + "\n"
    )
    if variant == "reason":
        notice += (
            "Previous fallible model analysis:\n"
            + json.dumps(
                dict(reason=row["previous_reason"], orientation=row["previous_orientation"]),
                ensure_ascii=False,
            )
            + "\n"
        )
    notice += "Check two specific questions before the final source-origin judgment: (1) Do the differing calls actually compute different required quantities or observable effects under corresponding inputs? Follow their operands and returned values; a common mathematical or string-processing purpose alone is insufficient. (2) Could the precise call difference be a supported lowering, inlining, specialization or library optimization that preserves the required operation? A missing named call or a different set of API names alone is not evidence of different source. For example, replacing a call with inline arithmetic or replacing formatted output with an equivalent simpler output can be valid. Do not invent an optimization that changes an essential formula, comparison semantics or a reachable output. Decide from the full pair; keep genuine matches with a concrete correspondence, and reject substantive differences.\n"
    return notice + plain
