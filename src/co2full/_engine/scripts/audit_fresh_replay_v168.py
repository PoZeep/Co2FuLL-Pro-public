import hashlib, json

from scripts.run_fresh_replay_v168 import read, decode, MODEL, RECOVERY, RECOVERY107, rationale


class Auditor:
    def execute(self, stage, rows, make_messages):
        folder = self.source / "stages" / stage
        originals = {r["pair_id"]: r for r in read(folder / "inputs_unlabeled.jsonl")}
        assert rows == originals, (stage, "inputs or eligibility mismatch")
        scheduled = read(folder / "requests_frozen.jsonl")
        scheduled += read(folder / "repair_requests_frozen.jsonl")
        reqs = (
            {(r["pair_id"], r["method"]): r for r in read(folder / "requests.jsonl")}
            if (folder / "requests.jsonl").exists()
            else {}
        )
        outputs = (
            {(r["pair_id"], r["method"]): r for r in read(folder / "outputs.jsonl")}
            if (folder / "outputs.jsonl").exists()
            else {}
        )
        assert len(scheduled) == len(reqs) == len(outputs)
        assert {r["pair_id"] for r in scheduled if not r["constrained"]} == set(rows)
        for j in scheduled:
            p = j["pair_id"]
            repair = j["constrained"]
            initial = stage in ["fewshot", "compile"]
            messages = make_messages(rows[p])
            if repair:
                messages = messages + [
                    dict(role="user", content=RECOVERY107 if stage == "reason107" else RECOVERY)
                ]
            expected = dict(model=MODEL, messages=messages, temperature=0, top_p=1)
            if not initial:
                expected["seed"] = 0
            if stage != "fewshot" or repair:
                expected["max_tokens"] = 16 if repair else 1024
            if repair:
                expected["guided_choice"] = ["yes", "no"]
            assert j["payload"] == expected, (stage, p, "prompt mismatch")
            req = reqs[p, j["method"]]
            out = outputs[p, j["method"]]
            assert all(req[k] == v for k, v in j.items())
            assert out["request_sha256"] == hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()
            response = req["response"]
            ch = response["choices"][0]
            assert response["model"] == MODEL and response["usage"]["prompt_tokens"] == j["prompt_tokens"]
            assert (
                out["raw_output"] == (ch["message"].get("content") or "")
                and out["usage"] == response["usage"]
                and out["finish_reason"] == ch["finish_reason"]
                and out["error"] is None
            )
            budget = 16 if repair else (1024 if initial else 2304)
            assert j["endpoint"] == (
                8074 if j["prompt_tokens"] + budget > 32768 else 8071 + int(p[:8], 16) % 3
            )
            assert j["prompt_tokens"] + budget <= 131072
            if repair:
                assert decode(outputs[p, stage]) is None, "Unnecessary repeated answer"
            self.checked += 1
        answers = {}
        for p in sorted(rows):
            o = outputs.get((p, stage + "_repair"), outputs[p, stage])
            pred = decode(o)
            assert pred in [0, 1]
            answers[p] = dict(
                pair_id=p,
                prediction=pred,
                raw_output=o["raw_output"],
                reason=rationale(o["raw_output"]),
                method=o["method"],
                constrained=o["constrained"],
            )
        assert answers == {r["pair_id"]: r for r in read(folder / "answers.jsonl")}
        self.stage_counts[stage] = dict(
            eligible=len(rows), done=len(rows), repairs=len(scheduled) - len(rows)
        )
        self.status(stage=stage, requests_checked=self.checked)
        return answers
