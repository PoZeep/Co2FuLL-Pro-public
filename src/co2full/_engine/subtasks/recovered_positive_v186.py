import json

from subtasks.methods import COMPILE_V2

from subtasks.compile_yes_v112 import ROLE_CONTRACT, CLAIM_WITNESS, PREFIX

from subtasks.grounded_facts_v184 import model_facts


def content(row, variant):
    a, b = row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"]
    if variant == "plain":
        return COMPILE_V2.format(code_a=a, code_b=b)
    context = (
        PREFIX
        + json.dumps(
            dict(
                previous_positive_analysis=row["previous_reason"],
                previous_analysis_orientation=row["previous_orientation"],
            ),
            ensure_ascii=False,
        )
        + "\n"
    )
    if row.get("facts"):
        context += (
            "Additional conditional facts previously supplied to the model; these do not prove whole-function origin:\n"
            + json.dumps(model_facts(row["facts"]), ensure_ascii=False)
            + "\n"
        )
    instruction = CLAIM_WITNESS if variant == "claim" else ROLE_CONTRACT
    return context + instruction + "\nCode A:\n" + a + "\n\nCode B:\n" + b
