from co2full.runtime import asset_root as _cf_asset_root

import copy

ROOT = _cf_asset_root()

from scripts.run_fresh_replay_v168 import Replay, read, write, lines, mm

from scripts.run_xm_prefilter_v100 import token_count

from scripts.run_background_cold_v181 import gates as gates181, content as content181

from scripts.run_background_tree_v182 import (
    Experiment as TreeEvidence,
    allowed as allowed182,
    prompt as prompt182,
)

from subtasks.grounded_facts_v184 import (
    return_facts,
    table_accesses,
    indexed_data_facts,
    read_elf,
    selftest as tests184,
    content as content184,
)

from subtasks.recovered_positive_v186 import content as content186

from subtasks.address_facts_v188 import evidence as notes188, selftest as tests188, content as content188

from subtasks.methods import COMPILE_V2

from subtasks.reason_consistency_v191 import eligible as gate191, content as prompt191

from subtasks.compiler_idioms_v194 import eligible as gate194, content as prompt194

from subtasks.helper_observations_v195 import content as prompt195

from subtasks.library_effects_v199 import eligible as gate199, content as prompt199

from scripts.live_helper_features_v203 import build as helper_features


class BaseComplete(Exception):
    pass


class Pipeline(Replay):
    def execute(self, stage, rows, make_messages):
        aa = self.backend(stage, rows, make_messages)
        for p, a in aa.items():
            ca, cb = [self.inputs[p][side]["pseudo_code"] for side in ["function_a", "function_b"]]
            msg = make_messages(rows[p])[-1]["content"]
            orient = {}
            for mode in ["find", "rfind"]:
                ia, ib = getattr(msg, mode)(ca), getattr(msg, mode)(cb)
                orient[mode] = ca != cb and ia >= 0 and ib >= 0 and ib < ia
                orient[mode + "_complete"] = ia >= 0 and ib >= 0
            self.actual_last[p] = dict(answer=a, **orient)
        return aa

    def snapshot(self, name, state):
        if self.auditing:
            expected = {
                x["pair_id"]: {k: v for k, v in x.items() if k != "pair_id"}
                for x in read(self.source / f"snapshots/{name}.jsonl")
            }
            assert state == expected, (name, "fresh terminal reconstruction mismatch")
        Replay.snapshot(self, name, state)
        if self.building_base and name == "strategy087":
            self.base087 = copy.deepcopy(state)
            raise BaseComplete()

    def transition(self, stage, rows, render, state, reverse=False):
        aa = self.execute(stage, rows, lambda r: mm(render(r)))
        self.update(state, aa)
        for p, a in aa.items():
            self.latest[p] = dict(answer=a, reverse=reverse)
        return aa

    def pipeline_anchor(self):
        self.building_base = True
        try:
            Replay.pipeline(self)
        except BaseComplete:
            pass
        else:
            raise AssertionError("Did not stop after the selected087 base")
        self.building_base = False
        state = copy.deepcopy(self.base087)
        structural = {x["pair_id"]: x for x in read(self.run / "static_routes_unlabeled.jsonl")}
        rows = {
            p: self.inputs[p]
            for p, x in self.base087.items()
            if gates181(x, structural[p], self.frozen)["expanded"]
        }
        self.transition("expanded181", rows, content181, state)
        self.snapshot("expanded181", state)
        potential = {
            p: self.inputs[p]
            for p, x in state.items()
            if allowed182(x["prediction"], x["source"], structural[p], self.frozen)
        }
        trees, _ = TreeEvidence.evidence(self, potential)
        raw = self.transition("tree_raw182", trees, lambda r: prompt182(r, "tree", "raw"), state)
        chosen = {p: trees[p] for p, a in raw.items() if a["prediction"] == 1}
        confirm = self.transition(
            "tree_confirm182", chosen, lambda r: prompt182(r, "tree", "confirm"), state, True
        )
        chosen = {p: trees[p] for p, a in confirm.items() if a["prediction"] == 1}
        double = self.transition(
            "tree_double182", chosen, lambda r: prompt182(r, "tree", "double"), state, True
        )
        self.snapshot("tree_double182", state)
        rows = {p: trees[p] for p, a in raw.items() if a["prediction"] == 1 and state[p]["prediction"] == 0}
        self.transition(
            "plain183",
            rows,
            lambda r: COMPILE_V2.format(
                code_a=r["function_a"]["pseudo_code"], code_b=r["function_b"]["pseudo_code"]
            ),
            state,
        )
        self.snapshot("plain183", state)
        facts = {}
        images = {}
        coverage = []
        write(self.run / "facts184_checks.json", tests184())
        for p, row in sorted(self.inputs.items()):
            a, b = [row[s]["pseudo_code"] for s in ["function_a", "function_b"]]
            ff = return_facts(a, b)
            if table_accesses(a) and table_accesses(b):
                keys = [row[s]["binary"] for s in ["function_a", "function_b"]]
                for k in keys:
                    if k not in images:
                        images[k] = read_elf(ROOT / "binaries" / k / "image")
                ff += indexed_data_facts(a, b, *[images[k] for k in keys])
            nt = (
                max(
                    token_count(self.tok, mm(content184(dict(row, facts=ff), v)))
                    for v in ["plain", "data", "swapped"]
                )
                if ff
                else None
            )
            ok = bool(ff and nt + 2304 <= 131072)
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                facts[p] = dict(row, facts=ff)
        lines(self.run / "facts184_inputs_unlabeled.jsonl", facts.values())
        lines(self.run / "facts184_coverage_unlabeled.jsonl", coverage)
        rows = {
            p: r
            for p, r in facts.items()
            if state[p]["source"] == "model"
            and (
                state[p]["prediction"] == 0
                or any(f["kind"] == "indexed_readonly_word_values" for f in r["facts"])
            )
        }
        self.transition("data184", rows, lambda r: content184(r, "data"), state)
        self.snapshot("data184", state)
        positive = {}
        coverage = []
        for p, x in sorted(state.items()):
            if not (x["source"] == "model" and x["prediction"] == 1 and self.base087[p]["prediction"] == 0):
                continue
            last = self.latest[p]
            assert last["answer"]["prediction"] == 1
            row = dict(
                self.inputs[p],
                previous_reason=last["answer"]["reason"],
                previous_orientation="Previous A is current B; previous B is current A."
                if last["reverse"]
                else "Previous A and B match current A and B.",
                facts=facts[p]["facts"] if p in facts else [],
            )
            nt = max(token_count(self.tok, mm(content186(row, v))) for v in ["plain", "claim", "role"])
            ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                positive[p] = row
        lines(self.run / "role186_coverage_unlabeled.jsonl", coverage)
        role = self.transition("role186", positive, lambda r: content186(r, "role"), state)
        self.snapshot("role186", state)
        rows = {}
        coverage = []
        observations = []
        write(self.run / "address188_checks.json", tests188())
        for p, a in sorted(role.items()):
            if a["prediction"] != 0:
                continue
            row = dict(self.inputs[p], negative_reason=a["reason"], facts=positive[p]["facts"])
            notes = notes188(row["function_a"]["pseudo_code"], row["function_b"]["pseudo_code"])
            row["address_notes"] = notes
            nt = max(token_count(self.tok, mm(content188(row, v))) for v in ["plain", "data", "reason"])
            ok = notes["available"] and nt + 2304 <= 131072
            coverage.append(
                dict(pair_id=p, available=ok, evidence_available=notes["available"], max_prompt_tokens=nt)
            )
            observations.append(dict(pair_id=p, address_notes=notes))
            if ok:
                rows[p] = row
        lines(self.run / "address188_coverage_unlabeled.jsonl", coverage)
        lines(self.run / "address188_notes_unlabeled.jsonl", observations)
        self.transition("plain188", rows, lambda r: content188(r, "plain"), state)
        self.snapshot("final188plain", state)
        self.anchor_state = copy.deepcopy(state)

    def reason_row(self, p, last, mode="find"):
        return dict(
            self.inputs[p],
            previous_reason=last["answer"]["reason"],
            previous_orientation="Previous A is current B; previous B is current A."
            if last[mode]
            else "Same A/B order.",
        )

    def pipeline_selected(self):
        self.pipeline_anchor()
        state = copy.deepcopy(self.anchor_state)
        last189 = copy.deepcopy(self.actual_last)
        rows = {}
        coverage = []
        for p, x in sorted(state.items()):
            if x["source"] != "model":
                continue
            a = last189[p]["answer"]
            assert a["prediction"] == x["prediction"]
            if not gate191(x["prediction"], x["source"], a["reason"]):
                continue
            row = self.reason_row(p, last189[p])
            nt = max(
                token_count(self.tok, mm(prompt191(row, v))) for v in ["plain", "reconcile", "invariant"]
            )
            ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                rows[p] = row
        lines(self.run / "coverage191_unlabeled.jsonl", coverage)
        self.transition("reconcile191", rows, lambda r: prompt191(r, "reconcile"), state)
        self.snapshot("reconcile191", state)
        rows = {}
        coverage = []
        for p, x in sorted(state.items()):
            if x["source"] != "model" or x["prediction"] != 0:
                continue
            last = last189[p]
            if not gate194(self.inputs[p], x["prediction"], x["source"], last["answer"]["reason"]):
                continue
            row = self.reason_row(p, last)
            nt = max(token_count(self.tok, mm(prompt194(row, v))) for v in ["plain", "idiom", "reason"])
            ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                rows[p] = row
        lines(self.run / "coverage194_unlabeled.jsonl", coverage)
        self.transition("idiom194", rows, lambda r: prompt194(r, "idiom"), state)
        self.snapshot("idiom194", state)
        state194 = copy.deepcopy(state)
        last194 = copy.deepcopy(self.actual_last)
        packets = helper_features(
            self.inputs,
            state194,
            self.run / "helper195",
            self.source / "helper195" if self.auditing else None,
        )
        rows = {}
        coverage = []
        for p, packet in sorted(packets.items()):
            if not packet["available"]:
                continue
            assert state194[p]["source"] == "model" and state194[p]["prediction"] == 0
            row = dict(self.inputs[p], observed_helpers=packet["observed_helpers"])
            nt = max(
                token_count(self.tok, mm(prompt195(row, v, r)))
                for v in ["plain", "compact", "annotated"]
                for r in [False, True]
            )
            ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                rows[p] = row
        lines(self.run / "coverage195_unlabeled.jsonl", coverage)
        raw = self.transition("compact195", rows, lambda r: prompt195(r, "compact"), state)
        chosen = {p: rows[p] for p, a in raw.items() if a["prediction"] == 1}
        self.transition("compact_confirm195", chosen, lambda r: prompt195(r, "compact", True), state, True)
        self.snapshot("compact_confirm195", state)
        rows = {}
        coverage = []
        for p, x in sorted(state194.items()):
            if not (x["source"] == "model" and x["prediction"] == 1 and gate199(self.inputs[p])):
                continue
            last = last194[p]
            assert last["answer"]["prediction"] == 1 and last["rfind_complete"]
            row = self.reason_row(p, last, "rfind")
            nt = max(
                token_count(self.tok, mm(prompt199(row, v, r)))
                for v in ["plain", "facts", "reason"]
                for r in [False, True]
            )
            ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_prompt_tokens=nt))
            if ok:
                rows[p] = row
        lines(self.run / "coverage199_unlabeled.jsonl", coverage)
        raw = self.transition("plain199", rows, lambda r: prompt199(r, "plain"), state)
        chosen = {p: rows[p] for p, a in raw.items() if a["prediction"] == 0}
        self.transition("plain_confirm199", chosen, lambda r: prompt199(r, "plain", True), state, True)
        self.snapshot("helper_pruned_prefix016", state)
        from subtasks.final_refinement_v011 import refine

        state = refine(self, state)
        self.snapshot("final_helper_pruned016", state)
