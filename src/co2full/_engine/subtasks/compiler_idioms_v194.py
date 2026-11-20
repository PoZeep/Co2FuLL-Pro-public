import re, json

from subtasks.methods import COMPILE_V2


def notes(row):
    texts = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
    result = []
    for side, text in zip(["A", "B"], texts):
        clean = re.sub(r"//[^\n]*|/\*.*?\*/", " ", text, flags=re.S)
        if (
            "0x8000000000000000" in clean
            and re.search(r">>\s*1\b", clean)
            and re.search(r"\b(?:double|float|fabs)\b|_f64", clean)
        ):
            result.append(
                dict(
                    side=side,
                    kind="unsigned_to_fp_candidate",
                    observation="High-bit test and right-shift-by-one in code with floating-point use.",
                    hypothesis="An unsigned-to-floating conversion may test the high bit, form (x>>1)|(x&1), convert that value, then double the floating result. A decompiler can show misleading integer assignments/casts. Check whether this feeds floating output or numerical work; an integer-only occurrence does not establish this interpretation.",
                )
            )
        if "_mm_unpacklo_epi32" in clean and "_mm_sub_pd" in clean:
            result.append(
                dict(
                    side=side,
                    kind="packed_integer_to_fp_candidate",
                    observation="Packed integer unpack followed by packed-double subtraction.",
                    hypothesis="This can be an integer-to-double conversion via exponent/bias construction, rather than a different vector algorithm. Its interpretation depends on the actual constants and operand roles. For a proven unsigned input, conversion preserves nonnegativity, so a negative-value test may disappear. Do not assume arbitrary vector constants have the required bias.",
                )
            )
        if "_mm_cvtsi32_si128" in clean and "_mm_or_si128" in clean:
            result.append(
                dict(
                    side=side,
                    kind="biased_integer_to_fp_candidate",
                    observation="Integer inserted into a SIMD register and bitwise OR used near numerical arithmetic.",
                    hypothesis="Combining integer bits with a floating-point exponent bias then subtracting the bias can convert an integer loop count to floating point. Check the data flow and constants; the OR need not be a source-level bitwise algorithm.",
                )
            )
        if (
            re.search(r">>\s*32\b", clean)
            and "%" in "\n".join(texts)
            and re.search(r"\b[0-9]{3,}\s*\*|\*\s*[0-9]{3,}\b", clean)
        ):
            result.append(
                dict(
                    side=side,
                    kind="constant_division_candidate",
                    observation="High-word shift and multiplication by a large literal; a remainder operation occurs in the function pair.",
                    hypothesis="Unsigned constant division/remainder can lower to a high-half multiply, corrective additions and shifts. A missing widening cast in pseudocode can hide the high-half operation. Compare candidate divisor, input width, quotient/remainder uses and guards. This is a pattern hypothesis, not proof that this exact sequence equals the other divisor.",
                )
            )
    if result:
        numbers = []
        for side, text in zip(["A", "B"], texts):
            clean = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', " ", text, flags=re.S)
            for m in re.finditer(r"(?<![\w.])(?:0x[0-9A-Fa-f]+|[0-9]+)(?:[uUlL]+)?(?![\w.])", clean):
                if (
                    not m.group().lower().startswith("0x")
                    and len(re.sub(r"[uUlL]+$", "", m.group())) > 1
                    and m.group().startswith("0")
                ):
                    continue
                value = (
                    int(re.sub(r"[uUlL]+$", "", m.group()), 0)
                    if m.group().lower().startswith("0x")
                    else int(re.sub(r"[uUlL]+$", "", m.group()))
                )
                if 256 <= value <= 2**64 - 1:
                    numbers.append((side, value))
        result.append(
            dict(
                kind="literal_notation",
                values=[dict(side=s, decimal=v, hex=hex(v)) for s, v in sorted(set(numbers))[:80]],
                scope="Notation conversion only; addresses, offsets, masks and numerical values can have different roles. Equal literals do not prove source identity.",
            )
        )
    return result


def eligible(row, prediction, source, reason):
    return source == "model" and prediction == 0 and bool(notes(row))


def content(row, variant):
    a, b = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
    prompt = COMPILE_V2.format(code_a=a, code_b=b)
    if variant == "plain":
        return prompt
    context = (
        "The following mechanically triggered notes identify POSSIBLE compiler idioms in the supplied code, not verified equivalence or a same-source label. Check their assumptions and reject an inapplicable hypothesis. Compare the actual complete computation and observable effects.\n"
        + json.dumps(notes(row), ensure_ascii=False)
        + "\n"
    )
    if variant == "reason":
        context += (
            "Previous fallible analysis and its orientation:\n"
            + json.dumps(
                dict(reason=row["previous_reason"], orientation=row["previous_orientation"]),
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        assert variant == "idiom"
    return context + prompt
