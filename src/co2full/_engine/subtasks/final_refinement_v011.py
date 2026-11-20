from collections import defaultdict

import copy, struct

from co2full.runtime import asset_root

from co2full.io import index, rows, write_rows

from scripts.run_fresh_replay_v168 import mm

from scripts.run_xm_prefilter_v100 import token_count

from subtasks.post_history_review_v011 import negative_gate, content as history_content

from subtasks.post_candidate_contrast_v011 import content as contrast_content

from subtasks.library_effects_v199 import eligible as api_asymmetry

ORDER = [
    "fewshot",
    "compile",
    "reason107",
    "type117",
    "type120",
    "flow121",
    "flow_confirm121",
    "positive124",
    "control134",
    "control_confirm134",
    "expanded181",
    "tree_raw182",
    "tree_confirm182",
    "tree_double182",
    "plain183",
    "data184",
    "role186",
    "plain188",
    "reconcile191",
    "idiom194",
    "compact195",
    "compact_confirm195",
    "plain199",
    "plain_confirm199",
]


def elf_facts(path):
    with path.open("rb") as handle:
        header = handle.read(64)
    if header[:4] != b"\x7fELF" or header[4] not in (1, 2) or header[5] not in (1, 2):
        return {"format": "unknown"}
    endian = "<" if header[5] == 1 else ">"
    machine = struct.unpack_from(endian + "H", header, 18)[0]
    return dict(
        format="ELF",
        bits=32 if header[4] == 1 else 64,
        endian="little" if header[5] == 1 else "big",
        machine={3: "x86", 8: "MIPS", 40: "ARM", 62: "x86-64", 183: "AArch64"}.get(machine, str(machine)),
    )


def refine(p, state):
    prefix = copy.deepcopy(state)
    origin = p.source if p.auditing else p.run
    histories = {}
    for stage in ORDER:
        folder = origin / "stages" / stage
        answers = index(folder / "answers.jsonl")
        for q in rows(folder / "requests_frozen.jsonl"):
            pid = q["pair_id"]
            a, b = [p.inputs[pid][s]["pseudo_code"] for s in ["function_a", "function_b"]]
            text = q["payload"]["messages"][-1]["content"]
            ia, ib = text.rfind(a), text.rfind(b)
            assert ia >= 0 and ib >= 0
            h = answers[pid]
            histories.setdefault(pid, []).append(
                dict(
                    stage=stage,
                    prediction=h["prediction"],
                    reason=h["reason"],
                    orientation="reversed" if a != b and ib < ia else "same",
                )
            )
    candidates = {"history_negative_v011": {}, "contrast_positive_v011": {}, "history_api_v011": {}}
    coverage = []

    def history_row(pid):
        history = histories[pid]
        old = prefix[pid]
        indices = {0, len(history) - 1}
        contrary = [i for i, h in enumerate(history) if h["prediction"] != old["prediction"]]
        if contrary:
            indices.add(contrary[-1])
        return dict(p.inputs[pid], history=[history[i] for i in sorted(indices)])

    def within(stage, pid, item, messages):
        nt = max(token_count(p.tok, messages(item, reverse)) for reverse in [False, True])
        ok = nt + 2304 <= 131072
        coverage.append(dict(stage=stage, pair_id=pid, available=ok, max_prompt_tokens=nt))
        if ok:
            candidates[stage][pid] = item

    hist_messages = lambda r, reverse: mm(history_content(r, reverse))
    contrast_messages = lambda r, reverse: (
        mm(contrast_content(r, reverse))[:1] + p.examples + mm(contrast_content(r, reverse))[1:]
    )
    for pid, old in sorted(prefix.items()):
        if old["source"] != "model":
            continue
        history = histories[pid]
        last = history[-1]
        assert last["prediction"] == old["prediction"]
        if old["prediction"] == 0 and negative_gate(last["reason"]):
            within("history_negative_v011", pid, history_row(pid), hist_messages)
        if (
            old["prediction"] == 1
            and history[0]["prediction"] == 0
            and len(history) >= 2
            and history[-2]["prediction"] == 1
            and last["orientation"] == "same"
            and api_asymmetry(p.inputs[pid])
        ):
            within("history_api_v011", pid, history_row(pid), hist_messages)
    first = index(origin / "stages/fewshot/answers.jsonl")
    groups = defaultdict(list)
    binary_facts = {}
    for pid, item in p.inputs.items():
        if prefix[pid]["prediction"]:
            groups[item["function_a"]["binary"], item["function_a"]["address"]].append(pid)
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for pid in sorted(ids):
            if prefix[pid]["source"] != "model" or first[pid]["prediction"] != 0:
                continue
            alternatives = sorted(
                (
                    q
                    for q in ids
                    if q != pid and (prefix[q]["source"] == "rule" or first[q]["prediction"] == 1)
                ),
                key=lambda q: (
                    -(
                        p.structure[q]["score"]["cosine"]
                        if p.structure[q]["score"]["cosine"] is not None
                        else float("-inf")
                    ),
                    q,
                ),
            )
            if not alternatives:
                continue
            other_id = alternatives[0]
            item = dict(p.inputs[pid], other=p.inputs[other_id]["function_b"], other_id=other_id)
            item["elf"] = {}
            for side, letter in [("function_a", "A"), ("function_b", "B"), ("other", "C")]:
                binary = item[side]["binary"]
                if binary not in binary_facts:
                    binary_facts[binary] = elf_facts(asset_root() / "binaries" / binary / "image")
                item["elf"][letter] = binary_facts[binary]
            within("contrast_positive_v011", pid, item, contrast_messages)
    write_rows(p.run / "post_refinement_v011_coverage.jsonl", coverage)
    outputs = {}
    for stage in ["history_negative_v011", "contrast_positive_v011", "history_api_v011"]:
        render = contrast_messages if stage == "contrast_positive_v011" else hist_messages
        outputs[stage] = p.execute(stage, candidates[stage], lambda r, render=render: render(r, False))
    final = copy.deepcopy(prefix)
    for pid, a in outputs["history_negative_v011"].items():
        if a["prediction"] == 1:
            final[pid] = dict(prediction=1, source="model", terminal="history_negative_v011")
    for stage in ["contrast_positive_v011", "history_api_v011"]:
        for pid, a in outputs[stage].items():
            if a["prediction"] == 0:
                final[pid] = dict(prediction=0, source="model", terminal=stage)
    assert set(final) == set(prefix)
    assert all(final[k] == v for k, v in prefix.items() if v["source"] == "rule")
    return final
