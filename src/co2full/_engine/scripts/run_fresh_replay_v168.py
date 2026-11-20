import copy, hashlib, json

import baseline_fewshot as base

from scripts.run_compile_recheck_v107 import decode

from scripts.run_xm_prefilter_v100 import messages as old_reason_messages, token_count

from subtasks.methods import COMPILE_V2

from subtasks.reason_review_v91 import rationale

from subtasks.structural_triage_v106 import route, decide

from subtasks import representation_review_v115 as type_review

from subtasks import confirmation_v119 as type_confirm

from subtasks import control_flow_rescue_v121 as flow_review

from subtasks import compile_positive_repeat_v124 as positive_review

from subtasks import dual_remaining_v134 as dual_review

SIDES = ["function_a", "function_b"]

MODEL = "Qwen2.5-7B-Instruct"

RECOVERY = "Give your final source-origin judgment on the complete supplied code now. Return exactly yes or no. Both choices are allowed; no markup or explanation."

RECOVERY107 = "The previous attempt did not finish with a readable final label. Judge the complete supplied code now. Return exactly yes or no. Both choices are allowed; no explanation or markup."


def read(p):
    with p.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def lines(p, xs):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def mm(text):
    return [dict(role="system", content=base.SYSTEM_PROMPT), dict(role="user", content=text)]


class Replay:
    def snapshot(self, name, state):
        assert state.keys() == self.inputs.keys() and all(v["prediction"] in [0, 1] for v in state.values())
        path = self.run / "snapshots" / (name + ".jsonl")
        lines(path, [dict(pair_id=p, **state[p]) for p in sorted(state)])
        self.snapshots[name] = path

    @staticmethod
    def update(state, answers):
        for p, a in answers.items():
            state[p] = dict(prediction=a["prediction"], source="model", terminal=a["method"])

    def dual_rows(self, ids):
        chosen = {}
        coverage = []
        for p in sorted(ids):
            row = self.inputs[p]
            alt = {}
            ok = True
            for s in SIDES:
                f = row[s]
                v = self.views.get((f["binary"], f["address"]), {})
                if v:
                    assert v["original"] == f["pseudo_code"]
                ok = ok and bool(v.get("usable") and v.get("ghidra"))
                alt[s] = v.get("ghidra", "")
            r = dict(row, alternate=alt)
            nt = None
            if ok:
                nt = max(
                    token_count(self.tok, mm(dual_review.content(r, m)))
                    for m in ["control_raw_1", "context_raw_1", "context_confirm_1"]
                )
                ok = nt + 2304 <= 131072
            coverage.append(dict(pair_id=p, available=ok, max_tokens=nt))
            if ok:
                chosen[p] = r
        lines(self.run / "dual_coverage.jsonl", coverage)
        return chosen

    def pipeline(self):
        routes = {
            p: route(s["score"], s["graph"], self.frozen, direct_no=True) for p, s in self.structure.items()
        }
        lines(
            self.run / "static_routes_unlabeled.jsonl",
            [dict(pair_id=p, **routes[p], **self.structure[p]) for p in sorted(routes)],
        )
        fsrows = {p: r for p, r in self.inputs.items() if routes[p]["action"] not in ["rule_yes", "rule_no"]}
        fs = self.execute(
            "fewshot", fsrows, lambda r: base.make_messages(r, self.examples, True, "pseudo_code")
        )
        crows = {
            p: self.inputs[p]
            for p in fs
            if fs[p]["prediction"] == 0 and routes[p]["action"] == "fewshot_then_compile"
        }
        cc = self.execute(
            "compile",
            crows,
            lambda r: mm(
                COMPILE_V2.format(
                    code_a=r["function_a"]["pseudo_code"], code_b=r["function_b"]["pseudo_code"]
                )
            ),
        )
        state = {}
        tri = {}
        for p in self.inputs:
            f = fs[p]["prediction"] if p in fs else None
            c = cc[p]["prediction"] if p in cc else None
            d = decide(routes[p], f, c)
            tri[p] = dict(
                pair_id=p,
                fewshot=f,
                compile=c,
                **self.structure[p],
                arms={"structural_yes_no_then_models": dict(routes[p], **d)},
            )
            state[p] = dict(
                prediction=d["prediction"],
                source="rule" if d["source"].startswith("rule") else "model",
                terminal=d["source"],
            )
        lines(self.run / "triage_unlabeled.jsonl", tri.values())
        self.snapshot("triage", state)
        rows = {
            p: dict(self.inputs[p], original=dict(prediction=fs[p]["prediction"], reason=fs[p]["reason"]))
            for p in cc
            if cc[p]["prediction"] == 0
        }
        r107 = self.execute("reason107", rows, lambda r: old_reason_messages(r, "old_reason_compile"))
        self.update(state, r107)
        self.snapshot("reason107", state)
        rows = {
            p: dict(self.inputs[p], previous_fewshot_reason=fs[p]["reason"])
            for p in fs
            if type_review.eligible(tri[p], fs[p]["reason"])
        }
        r117 = self.execute("type117", rows, lambda r: mm(type_review.content(r, "control_1")))
        r120 = self.execute(
            "type120",
            {p: rows[p] for p, a in r117.items() if a["prediction"] == 1},
            lambda r: mm(type_confirm.content(r, "blind_reverse_1")),
        )
        self.update(state, r117)
        self.update(state, r120)
        self.snapshot("type120", state)
        rows = {
            p: dict(self.inputs[p], previous_fewshot_reason=fs[p]["reason"])
            for p in fs
            if flow_review.eligible(tri[p], fs[p]["reason"])
        }
        r121 = self.execute("flow121", rows, lambda r: mm(flow_review.content(r, "reason_raw_1")))
        r121c = self.execute(
            "flow_confirm121",
            {p: rows[p] for p, a in r121.items() if a["prediction"] == 1},
            lambda r: mm(flow_review.content(r, "reason_confirm_1")),
        )
        self.update(state, r121)
        self.update(state, r121c)
        self.snapshot("flow121", state)
        rows = {
            p: dict(self.inputs[p], previous_compile_reason=(r107[p] if p in r107 else cc[p])["reason"])
            for p in cc
            if state[p]["prediction"] == 1
        }
        r124 = self.execute("positive124", rows, lambda r: mm(positive_review.content(r, "review_1")))
        self.update(state, r124)
        self.snapshot("positive124", state)
        state124 = copy.deepcopy(state)
        rows = self.dual_rows([p for p in cc if cc[p]["prediction"] == 0 and state[p]["prediction"] == 0])
        control = self.execute("control134", rows, lambda r: mm(dual_review.content(r, "control_raw_1")))
        controlc = self.execute(
            "control_confirm134",
            {p: rows[p] for p, a in control.items() if a["prediction"] == 1},
            lambda r: mm(dual_review.content(r, "control_confirm_1")),
        )
        state087 = copy.deepcopy(state)
        self.update(state087, control)
        self.update(state087, controlc)
        self.snapshot("strategy087", state087)
        return state087
