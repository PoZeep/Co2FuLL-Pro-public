import json

INSTRUCTION = """Check whether TARGET and QUERY originate from the same source function under different compilation settings. OTHER is another retrieved candidate supplied as contrastive code evidence, not ground truth. Either, both, or neither candidate may match the query. Do not choose a winner just because another candidate looks closer.

First identify the query's distinctive operation: its exact update formula, condition, returned quantity, and persistent writes. Compare how TARGET and OTHER implement that operation. The additional code may reveal that a superficially shared template has a different input role, comparison direction, mathematical quantity, or state effect. Common library calls or nearly identical scaffolding alone are insufficient.
Then decide QUERY versus TARGET only. Reject TARGET only for a specific source-level incompatibility supported by the code, not because OTHER is ranked or preferred. Preserve a match when differences are explained by actual ABI width, structure layout, inlining, vectorization, strength reduction, or decompiler output recovery. Byte offsets must be compared after accounting for pointer sizes and field layouts. Inferred temporary types and different anonymous names do not prove different source. Unknown helpers remain unknown; do not invent their operations. Ignore function names as identity shortcuts.
Explain in at most 200 words: the query's distinguishing operation, TARGET's corresponding operation, and whether a real incompatible computation remains. Conclude exactly <same_source>yes</same_source> or <same_source>no</same_source>, for QUERY versus TARGET. The presence of OTHER never requires a NO.
"""


def content(row, reverse=False):
    blocks = [
        ("QUERY", row["function_a"]["pseudo_code"], row["elf"]["A"]),
        ("TARGET", row["function_b"]["pseudo_code"], row["elf"]["B"]),
        ("OTHER", row["other"]["pseudo_code"], row["elf"]["C"]),
    ]
    if reverse:
        blocks = [blocks[1], blocks[0], blocks[2]]
    return INSTRUCTION + "".join(
        "\n\n" + name + " — ELF header facts " + json.dumps(facts) + ":\n" + code
        for name, code, facts in blocks
    )
