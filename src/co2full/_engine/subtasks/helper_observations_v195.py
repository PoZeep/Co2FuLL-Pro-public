import copy, json, re

from subtasks.methods import COMPILE_V2

NOTICE = """Finite machine-code probes identified the anonymous helper operations listed below under stated register ABI hypotheses. These are empirical observations (initial probes plus 24 additional cases, two unspecified-register settings), not proofs. The decompiler can omit or mistype helper arguments and return words. Do not interpret a missing displayed argument as a missing mathematical input. Caller-argument mapping is NOT established by these probes. Check the actual loop, memory access, operand flow and returned quantity; shared generic arithmetic alone does not establish common source. Unknown calls remain unknown.\n"""


def records(row, side):
    result = []
    text = row[side]["pseudo_code"]
    for h in row["observed_helpers"].get(side, []):
        addr = int(h["address"], 16)
        names = sorted(
            {m[1] for m in re.finditer(r"\b((?:j_)?sub_([0-9a-fA-F]+))\s*\(", text) if int(m[2], 16) == addr}
        )
        for name in names:
            result.append(dict(callee=name, operation=h["primitive"], register_ABI=h["abi"]))
    return result


def content(row, variant, reverse=False):
    if reverse:
        row = copy.deepcopy(row)
        row["function_a"], row["function_b"] = row["function_b"], row["function_a"]
        row["observed_helpers"] = {
            ("function_b" if s == "function_a" else "function_a"): v
            for s, v in row["observed_helpers"].items()
        }
    a, b = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
    if variant == "plain":
        return COMPILE_V2.format(code_a=a, code_b=b)
    assert variant in ["compact", "annotated"]
    tables = {s: records(row, s) for s in ["function_a", "function_b"]}
    if variant == "annotated":
        texts = []
        for side, text in zip(["function_a", "function_b"], [a, b]):
            for h in tables[side]:
                text = re.sub(
                    r"\b" + re.escape(h["callee"]) + r"(?=\s*\()",
                    h["callee"] + " /* observed " + h["operation"] + "; " + h["register_ABI"] + " */",
                    text,
                )
            texts.append(text)
        a, b = texts
        notes = "Comments immediately before anonymous calls were added from the observations; all other original code is retained.\n"
    else:
        notes = json.dumps({"A": tables["function_a"], "B": tables["function_b"]}, ensure_ascii=False) + "\n"
    return NOTICE + notes + COMPILE_V2.format(code_a=a, code_b=b)
