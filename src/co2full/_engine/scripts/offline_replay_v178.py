from concurrent.futures import ThreadPoolExecutor

import hashlib, json, random, threading, time, urllib.request

import baseline_fewshot as base

from scripts.run_fresh_replay_v168 import (
    MODEL,
    RECOVERY,
    RECOVERY107,
    decode,
    rationale,
    token_count,
    read,
    write,
    lines,
    sha,
)


class OfflineReplay:
    def execute(self, stage, rows, make_messages):
        ids = sorted(rows)
        index = len(self.stage_counts)
        self.stage_counts[stage] = dict(eligible=len(ids), done=0)
        self.status(stage=stage, total=len(ids), done=0)
        folder = self.run / "stages" / stage
        folder.mkdir(parents=True, exist_ok=True)
        lines(folder / "inputs_unlabeled.jsonl", [rows[p] for p in ids])

        baseline = stage == "fewshot"
        initial = stage in ["fewshot", "compile"]

        def job(p, repair=False):
            messages = make_messages(rows[p])
            if repair:
                messages = messages + [
                    dict(role="user", content=RECOVERY107 if stage == "reason107" else RECOVERY)
                ]
            nt = token_count(self.tok, messages)
            budget = 16 if repair else (1024 if initial else 2304)
            assert nt + budget <= 131072, (p, stage, nt)
            payload = dict(model=MODEL, messages=messages, temperature=0, top_p=1)
            if not initial:
                payload["seed"] = 0
            if not baseline or repair:
                payload["max_tokens"] = 16 if repair else 1024
            if repair:
                payload["guided_choice"] = ["yes", "no"]
            return dict(
                pair_id=p,
                method=stage + ("_repair" if repair else ""),
                prompt_tokens=nt,
                endpoint=8074 if nt + budget > 32768 else 8071 + int(p[:8], 16) % 3,
                constrained=repair,
                payload=payload,
            )

        jobs = [job(p) for p in ids]
        lines(folder / "requests_frozen.jsonl", jobs)
        write(
            folder / "requests_fixed.json",
            dict(unix_time=time.time(), sha256=sha(folder / "requests_frozen.jsonl")),
        )
        done = (
            {o["method"] + ":" + o["pair_id"]: o for o in read(folder / "outputs.jsonl")}
            if (folder / "outputs.jsonl").exists()
            else {}
        )
        width = 1
        limits = {p: threading.Semaphore(1 if p == 8074 else width) for p in range(8071, 8075)}

        def invoke(j):
            req = dict(j, attempts=[])
            out = {k: j[k] for k in ["pair_id", "method", "prompt_tokens", "endpoint", "constrained"]}
            out["error"] = None
            with limits[j["endpoint"]]:
                for attempt in range(2):
                    tick = time.time()
                    try:
                        request = urllib.request.Request(
                            f"http://127.0.0.1:{j['endpoint']}/v1/chat/completions",
                            data=json.dumps(j["payload"]).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=1800) as h:
                            response = json.load(h)
                    except Exception as e:
                        req["attempts"].append(dict(error=repr(e), seconds=time.time() - tick))
                        if attempt == 0:
                            time.sleep(2)
                            continue
                        out["error"] = repr(e)
                    else:
                        req["attempts"].append(dict(seconds=time.time() - tick))
                        req["response"] = response
                        choice = response["choices"][0]
                        assert (
                            response["model"] == MODEL
                            and response["usage"]["prompt_tokens"] == j["prompt_tokens"]
                        )
                        out.update(
                            raw_output=choice["message"].get("content") or "",
                            finish_reason=choice["finish_reason"],
                            usage=response["usage"],
                        )
                    break
            out["request_sha256"] = hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()
            return out, req

        def batch(batch_jobs, phase):
            pending = []
            for j in batch_jobs:
                key = j["method"] + ":" + j["pair_id"]
                if key in done:
                    assert (
                        done[key]["request_sha256"]
                        == hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()
                    )
                    assert done[key]["error"] is None
                else:
                    pending.append(j)
            random.Random(17800).shuffle(pending)

            def endpoint_batches(port, selected):
                block = 1 if port == 8074 else 32
                for start in range(0, len(selected), block):
                    chunk = selected[start : start + block]
                    if port == 8074:
                        yield [invoke(chunk[0])]
                        continue
                    tick = time.time()
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{port + 100}/batch",
                        data=json.dumps(dict(jobs=chunk)).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=3600) as h:
                        adapted = json.load(h)
                    assert adapted["adapter"] == "offline_v178" and len(adapted["records"]) == len(chunk)
                    result = []
                    for j, z in zip(chunk, adapted["records"]):
                        assert (z["pair_id"], z["method"]) == (j["pair_id"], j["method"])
                        response = z["response"]
                        engine = z["engine_output"]
                        ch = response["choices"][0]
                        assert response["model"] == MODEL and response["usage"]["prompt_tokens"] == j[
                            "prompt_tokens"
                        ] == len(engine["prompt_token_ids"])
                        assert response["usage"]["completion_tokens"] == len(engine["output_token_ids"])
                        assert (
                            ch["message"]["content"] == engine["text"]
                            and ch["finish_reason"] == engine["finish_reason"]
                        )
                        req = dict(
                            j,
                            attempts=[dict(seconds=time.time() - tick)],
                            response=response,
                            transport="offline_v178_adapter",
                            transport_endpoint=port + 100,
                            engine_output=engine,
                            effective=z["effective"],
                            offline_batch=dict(
                                number=adapted["batch_number"],
                                index=start // block,
                                ordered_pairs=[x["pair_id"] for x in chunk],
                                worker_pid=adapted["pid"],
                            ),
                        )
                        out = {
                            k: j[k] for k in ["pair_id", "method", "prompt_tokens", "endpoint", "constrained"]
                        }
                        out.update(
                            error=None,
                            raw_output=engine["text"],
                            finish_reason=engine["finish_reason"],
                            usage=response["usage"],
                            request_sha256=hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest(),
                        )
                        result.append((out, req))
                    yield result

            import queue

            events = queue.Queue()

            def consume(port):
                try:
                    selected = [j for j in pending if j["endpoint"] == port]
                    for batch_result in endpoint_batches(port, selected):
                        events.put(("batch", batch_result))
                except BaseException as e:
                    events.put(("error", e))
                finally:
                    events.put(("end", port))

            with ThreadPoolExecutor(max_workers=1) as pool:
                futures = [pool.submit(consume, port) for port in range(8071, 8075)]
                ended = 0
                failure = None
                while ended < 4:
                    kind, value = events.get()
                    if kind == "end":
                        ended += 1
                        continue
                    if kind == "error":
                        failure = value
                        continue
                    for out, req in value:
                        self.calls += 1
                        base.append_jsonl(folder / "requests.jsonl", req)
                        base.append_jsonl(folder / "outputs.jsonl", out)
                        done[out["method"] + ":" + out["pair_id"]] = out
                        self.stage_counts[stage].update(
                            done=sum(stage + ":" + p in done for p in ids),
                            repair_done=sum(stage + "_repair:" + p in done for p in ids),
                        )
                        self.status(
                            stage=stage, phase=phase, total=len(ids), done=self.stage_counts[stage]["done"]
                        )
                        assert out["error"] is None, (stage, out["pair_id"], out["error"])
                if failure:
                    raise failure

        batch(jobs, "inference")
        repairs = [job(p, True) for p in ids if decode(done[stage + ":" + p]) is None]
        lines(folder / "repair_requests_frozen.jsonl", repairs)
        if repairs:
            batch(repairs, "format_recovery")
        answers = {}
        for p in ids:
            o = done.get(stage + "_repair:" + p, done[stage + ":" + p])
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
        lines(folder / "answers.jsonl", answers.values())
        self.stage_counts[stage].update(
            done=len(ids),
            repairs=len(repairs),
            prompt_tokens=sum(o["usage"]["prompt_tokens"] for o in done.values()),
            completion_tokens=sum(o["usage"]["completion_tokens"] for o in done.values()),
        )
        write(
            folder / "complete.json",
            dict(
                unix_time=time.time(),
                answers_sha256=sha(folder / "answers.jsonl"),
                **self.stage_counts[stage],
            ),
        )
        self.status(stage=stage, phase="complete")
        print(json.dumps(dict(stage=stage, **self.stage_counts[stage])), flush=True)
        return answers
