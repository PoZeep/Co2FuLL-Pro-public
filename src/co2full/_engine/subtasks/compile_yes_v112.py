import json

from subtasks.methods import COMPILE_V2

PREFIX = (
    "The previous analysis is fallible auxiliary material, not a true label. "
    "Use the complete code to make your own source-origin decision. "
    "An incorrect old explanation alone does not establish either answer.\n"
)

ROLE_CONTRACT = """Review whether the two complete decompiled functions can be builds of the same original source function. The previous positive analysis may have mistaken common purpose for a specific source correspondence.
Work with concrete code evidence, not an algorithm name:
1. Map the roles of externally supplied inputs and the observable outputs. Outputs include returned values, output-parameter writes, updates to persistent state and effects of calls. A changed parameter count or inferred return type is not by itself a changed contract: account for aggregate-return ABI, pointer width and supported constant specialization.
2. Compare the distinctive computation through that mapping. Track what is read, what is written, and which values control those effects. Quote one short relevant expression from each side. Normalize equivalent literal values and branch spellings before treating them as differences. Do not assume unknown helpers are either identical or different.
3. Check the strongest claimed compilation explanation. An explanation must account for the actual computation and effects, not just rename the purpose. A reachable difference in input element interpretation, branch outcome, returned result or essential write can contradict a proposed correspondence. But a difference in spelling, an opaque helper, incomplete decompiler information, or lack of a formal proof alone does not contradict common source. Do not invent missing writes, removed behavior or a convenient optimization.
Keep a supported positive correspondence unless you find a concrete substantive incompatibility. State uncertainties without manufacturing a difference. Give your source-origin judgment, not an instruction-level equivalence verdict.
In at most250 words state the role mapping, the two short code quotations and the check. Finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source>.
"""

CLAIM_WITNESS = """Audit ONE decisive code claim in the previous positive source-origin analysis, then give a final source-origin judgment for the two complete functions.
Choose the claim whose failure would most undermine the old YES: an asserted corresponding operation, output, side effect, branch behavior or proposed compilation transformation. Do not choose mere anonymous names or an inferred type spelling when a substantive claim is available.
Write four short items:
- CLAIM: the precise assertion you are checking.
- A / B: one short exact code excerpt from each side that bears on that assertion. Read surrounding conditions and data dependencies in the complete functions.
- CHECK: track the relevant value or effect. If you propose a distinguishing input/path, it must use corresponding logical input roles and respect the shown conditions. Do not pretend a hypothetical trace was executed. Recognize equal constants, word packing, specialization, inlining and return-ABI differences when supported; do not assume a helper's unknown behavior. Explain why a remaining difference is observable, or why the proposed compiler explanation accounts for it.
- DECISION: integrate this check with the actual distinctive computation. Shared purpose alone is weak, but uncertainty or an error in the old rationale is not sufficient to reject matching code. Retain YES when a specific correspondence survives; revise to NO only for a concrete incompatible operation or behavior supported by the code. Do not require a formal equivalence proof.
Use at most250 words. Finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source>.
"""


def content(row, arm):
    variant = arm.rsplit("_", 1)[0]
    previous = (
        PREFIX
        + json.dumps(dict(previous_compile_explanation=row["previous_compile_reason"]), ensure_ascii=False)
        + "\n\n"
    )
    if variant == "control":
        return previous + COMPILE_V2.format(
            code_a=row["function_a"]["pseudo_code"], code_b=row["function_b"]["pseudo_code"]
        )
    instruction = {"role_contract": ROLE_CONTRACT, "claim_witness": CLAIM_WITNESS}[variant]
    return (
        previous
        + instruction
        + "\nCode A:\n"
        + row["function_a"]["pseudo_code"]
        + "\n\nCode B:\n"
        + row["function_b"]["pseudo_code"]
    )
