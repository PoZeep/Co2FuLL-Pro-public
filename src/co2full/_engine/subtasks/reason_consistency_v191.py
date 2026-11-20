import re, json

from subtasks.methods import COMPILE_V2


def eligible(prediction, source, reason):
    if source != "model" or prediction != 0 or not reason:
        return False
    tail = reason[-1100:].lower()
    return bool(
        re.search(r"(?:compil|decompil)", tail)
        and (
            re.search(r"rather than[^.\n]{0,160}(?:different|changes?|fundamental)", tail)
            or re.search(r"not (?:due to|changes? in)[^.\n]{0,120}(?:source|algorithm)", tail)
            or re.search(
                r"suggests? (?:that )?(?:they|these|the functions) are (?:different )?builds of the same",
                tail,
            )
        )
    )


RECONCILE = """Independently judge the source origin of the two complete decompiled functions. A previous NO was accompanied by the following fallible explanation. Its final discussion may attribute the differences to compilation while its binary answer rejects common source. This is a reason to CHECK, not a reason to automatically flip the answer.
First state what the explanation actually claims about source identity. Then check the distinctive computation against the code: align logical inputs, returned value, output writes, main update formula and branch effects. Distinguish representation/ABI differences from a changed algorithm. Unknown helper addresses and inferred integer versus floating types do not establish different source: values may be packed words and calls may implement arithmetic. Conversely, common purpose and a compiler keyword do not establish same source. Require a specific correspondence, not an invented optimization.
Resolve the conflict using the code. Ignore the old binary verdict if the code and supported reasoning contradict it; retain NO when there is a substantive source difference. Do not claim to execute code or assume missing helper semantics. Explain in at most 240 words and finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source>.
"""

INVARIANT = """Compare source origin through the algorithm's invariants, allowing compilation and decompilation differences. Read both complete functions before answering.
Write three compact items: (1) corresponding logical inputs and outputs, including ABI-packed values and output parameters; (2) the distinctive per-iteration update or effect sequence in both functions; (3) whether the differences affect that computation or only its representation. Normalize indexing to logical elements, loop direction with its accessed sequence, equivalent branches, inlining, SIMD lanes, and software versus hardware arithmetic when supported. Do not equate anonymous call names across binaries or treat different addresses as different semantics. Do not infer different source solely from apparent integer types where values feed floating arithmetic or split-word helpers. Do not excuse different mathematical formulas, meaningful constants or externally observable effects as optimization without evidence.
The task is common source origin, not exact machine-level equivalence. Similar purpose alone is insufficient; incomplete decompiler information alone is not disproof. Use at most 240 words and finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source>.
"""


def content(row, variant):
    a, b = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
    if variant == "plain":
        return COMPILE_V2.format(code_a=a, code_b=b)
    if variant == "reconcile":
        instruction = (
            RECONCILE
            + "\nPrevious explanation and orientation:\n"
            + json.dumps(
                dict(reason=row["previous_reason"], orientation=row["previous_orientation"]),
                ensure_ascii=False,
            )
        )
    else:
        assert variant == "invariant"
        instruction = INVARIANT
    return instruction + "\nCode A:\n" + a + "\n\nCode B:\n" + b
