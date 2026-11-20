import json

import re

from subtasks.methods import COMPILE_V2

TRIGGER = re.compile(
    r"\b(?:types?|signature|offsets?|width|32.bit|64.bit|signed|unsigned|pointers?|arguments?|parameters?|ABI)\b",
    re.I,
)

REVIEW = """Decide whether these complete decompiled functions can originate from the same source function. Audit the old negative explanation specifically for confusing machine representation with source behavior.
Use this short procedure; do not rewrite either function:
1. Identify the concrete type, argument, pointer, offset, or return representation difference actually alleged by the old explanation. Locate the relevant operations in BOTH functions. An allegation unsupported by the code is not a real difference.
2. Try a consistent correspondence of logical input roles and memory objects. Pointer width, element scaling, aggregate-return ABI, sign/zero extension, and compiler-created temporaries can change apparent types or signatures. Decimal and hexadecimal spellings can express the same value. A valid explanation must account for ALL affected accesses and uses, not just one coincidentally matching offset. Do not assert a particular architecture, structure layout, or helper behavior without evidence. Do not equate arbitrary offsets or pretend a trace was executed.
3. Check what remains after that correspondence: which inputs determine each return, persistent write, output-parameter write, and essential call effect? Determine whether there is a distinctive matching computation, or a real incompatible operation. State one strongest piece of code evidence from each side. Similar cleanup, allocation, looping, or search purpose alone does not establish common source; an error in the old reason also does not establish YES.
Output your own final source-origin judgment. Use YES if a specific computation and its effects correspond despite representation changes. Use NO if the functions implement different computations or the proposed correspondence relies only on generic purpose. An inferred type spelling, anonymous call address, or absence of a formal proof alone is not a substantive contradiction.
Keep the reasoning within250 words. Finish with exactly one <same_source>yes</same_source> or <same_source>no</same_source>.
"""


def eligible(route, reason):
    old = route["arms"]["structural_yes_no_then_models"]
    return (
        old["cause"] == "density_weak_compile_disabled"
        and route["fewshot"] == 0
        and bool(TRIGGER.search(reason))
    )


def content(row, arm):
    variant = arm.rsplit("_", 1)[0]
    previous = (
        "The previous negative analysis is fallible auxiliary material, not a true label. "
        "Use the complete code for your own source-origin judgment.\n"
        + json.dumps({"previous_fewshot_explanation": row["previous_fewshot_reason"]}, ensure_ascii=False)
        + "\n\n"
    )
    a, b = row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"]
    if variant == "control":
        return previous + COMPILE_V2.format(code_a=a, code_b=b)
    assert variant == "representation"
    return previous + REVIEW + "\nCode A:\n" + a + "\n\nCode B:\n" + b
