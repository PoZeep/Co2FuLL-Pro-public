"""Evaluation metrics and run inspection utilities."""

from pathlib import Path
from collections import Counter
from .io import index, read_json, rows, sha, write_json, write_rows
from .runtime import config, RESOURCES


def metrics(labels, predictions, retrieval_misses=0):
    if type(retrieval_misses) is not int or retrieval_misses < 0:
        raise ValueError("retrieval_misses must be a nonnegative integer")
    if any(type(x) is not int or x not in (0, 1) for x in (*labels.values(), *predictions.values())):
        raise ValueError("Predictions and labels must be binary integers")
    if set(labels) != set(predictions):
        raise ValueError("Evaluation membership mismatch; never drop missing predictions")
    counts = Counter(
        ("tp" if predictions[p] else "fn") if y else ("fp" if predictions[p] else "tn")
        for p, y in labels.items()
    )
    tp, fp, fn, tn = [counts[k] for k in ["tp", "fp", "fn", "tn"]]
    fn += retrieval_misses
    return dict(
        pairs=len(labels),
        retrieval_misses=retrieval_misses,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f2=5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.0,
    )


def evaluate(run, labels_path, retrieval_misses=0, without_pv=False):
    from .pipeline import check_seal

    run = Path(run).resolve()
    check_seal(run, 1)
    check_seal(run, 2)
    if not without_pv:
        check_seal(run, 3)
        seal = read_json(run / "decisions_fixed_before_evaluation.json")
        if sha(run / "protocol.json") != seal["protocol_sha256"]:
            raise ValueError("Protocol seal mismatch")
        if sha(run / "predictions.jsonl") != seal["predictions_sha256"]:
            raise ValueError("Prediction seal mismatch")
    labels_rows = index(labels_path)
    if any(type(r.get("label")) is not int or r["label"] not in [0, 1] for r in labels_rows.values()):
        raise ValueError("Labels must be integers 0/1")
    labels = {p: r["label"] for p, r in labels_rows.items()}
    results = {}
    routing = index(run / "static_routes_unlabeled.jsonl")
    fs = index(run / "stages/fewshot/answers.jsonl")
    after_fs = {
        p: (1 if r["action"] == "rule_yes" else 0 if r["action"] == "rule_no" else fs[p]["prediction"])
        for p, r in routing.items()
    }
    results["routing_plus_fewshot"] = metrics(labels, after_fs, retrieval_misses)
    if without_pv:
        return dict(stage="routing_plus_fewshot", final=results["routing_plus_fewshot"])
    for stage, digest in seal["snapshots"].items():
        path = run / "snapshots" / (stage + ".jsonl")
        if sha(path) != digest:
            raise ValueError(f"Snapshot changed: {stage}")
        results[stage] = metrics(
            labels, {p: r["prediction"] for p, r in index(path).items()}, retrieval_misses
        )
    final = index(run / "predictions.jsonl")
    result = dict(
        state="EVALUATED",
        final=metrics(labels, {p: r["prediction"] for p, r in final.items()}, retrieval_misses),
        stages=results,
        module1_actions=dict(Counter(r["action"] for r in routing.values())),
        labels_sha256=sha(labels_path),
        prediction_sha256=sha(run / "predictions.jsonl"),
        inference_summary=read_json(run / "module3.json"),
    )
    out = run / "evaluation"
    out.mkdir(exist_ok=True)
    write_json(out / "metrics.json", result)
    write_rows(
        out / "errors.jsonl",
        [
            dict(pair_id=p, label=labels[p], **{k: v for k, v in r.items() if k != "pair_id"})
            for p, r in final.items()
            if r["prediction"] != labels[p]
        ],
    )
    table = [
        "# Reproduction metrics",
        "",
        "| Stage | Precision | Recall | F2 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in results.items():
        table.append(
            f"| {name} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f2']:.6f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    (out / "README.md").write_text("\n".join(table) + "\n")
    return result


def annotate_routes(run):
    """Annotate stage routing records with evidence availability details."""
    run = Path(run)
    result = []
    targets = (
        index(run / "helper_evidence/targets_unlabeled.jsonl")
        if (run / "helper_evidence/targets_unlabeled.jsonl").exists()
        else {}
    )
    packets = (
        index(run / "helper195/packets_unlabeled.jsonl")
        if (run / "helper195/packets_unlabeled.jsonl").exists()
        else {}
    )
    export_cache = {}
    for spec in [dict(stage="fewshot")] + read_json(RESOURCES / "stages.json"):
        stage = spec["stage"]
        file = run / "routing" / (stage + ".jsonl")
        if not file.exists():
            continue
        for r in rows(file):
            d = dict(r)
            p = r["pair_id"]
            cov = r.get("evidence_check") or {}
            if not r["called"]:
                if stage == "helper132" and p in targets:
                    target = targets[p].get("target")
                    if not target:
                        d["reason"] = "no_recognized_direct_helper"
                    else:
                        binary = target["binary"]
                        if binary not in export_cache:
                            export_cache[binary] = read_json(
                                run / "helper_evidence/exports" / (binary + ".json")
                            )
                        helper = export_cache[binary].get(target["address"], {})
                        if helper.get("status") != "ok" or not helper.get("code"):
                            d["reason"] = "helper_export_unavailable"
                            d["export_reason"] = helper.get("reason", helper.get("error"))
                        elif len(helper["code"]) > 8000:
                            d["reason"] = "helper_body_exceeds_8000_characters"
                elif stage == "compact195" and p in packets and not packets[p]["available"]:
                    d["reason"] = "no_unique_validated_helper_observation"
                elif cov:
                    if cov.get("reason"):
                        d["reason"] = cov["reason"]
                    elif cov.get("evidence_available") is False:
                        d["reason"] = "address_evidence_unavailable"
                    else:
                        n = cov.get("max_prompt_tokens", cov.get("max_tokens"))
                        if n is not None and n + 2304 > 131072:
                            d["reason"] = "context_budget_exceeded"
                        elif stage == "control134" and n is None:
                            d["reason"] = "bilateral_ghidra_view_unavailable"
                        elif stage == "data184" and n is None:
                            d["reason"] = "no_supported_readonly_or_return_facts"
                        else:
                            d["reason"] = "required_evidence_unavailable"
            d["annotation_basis"] = "recorded candidate/coverage/helper artifacts; no labels"
            result.append(d)
    return result


def inspect(run, pair_id=None, out=None, assets_override=None):
    run = Path(run).resolve()
    if not pair_id:
        result = dict(run=str(run), status=read_json(run / "status.json"))
        result["completed_modules"] = [n for n in [1, 2, 3] if (run / f"module{n}.json").exists()]
        if (run / "evaluation/metrics.json").exists():
            result["metrics"] = read_json(run / "evaluation/metrics.json")["final"]
        if out:
            target = Path(out).resolve()
            target.mkdir(parents=True, exist_ok=False)
            annotated = annotate_routes(run)
            write_rows(target / "routing-details.jsonl", annotated)
            counts = Counter((r["stage"], r["reason"]) for r in annotated)
            write_json(
                target / "routing-summary.json",
                [dict(stage=s, reason=reason, count=n) for (s, reason), n in sorted(counts.items())],
            )
            write_json(target / "run-summary.json", result)
            result["report"] = str(target)
        return result
    if not out:
        raise ValueError("--out is required for a full pair export")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = config(run / "machine.json")
    asset_dir = Path(assets_override).expanduser().resolve() if assets_override else Path(cfg["assets"])
    input_path = asset_dir / "pairs.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(
            "Pair inputs are unavailable here; pass --assets /local/path/to/imported-assets. Do not modify the sealed machine.json."
        )
    if sha(input_path) != read_json(run / "protocol.json")["input_sha256"]:
        raise ValueError("Input assets differ from this run")
    pairs = index(input_path)
    if pair_id not in pairs:
        raise KeyError(f"Unknown pair: {pair_id}")
    write_json(out / "input.json", pairs[pair_id])
    route = index(run / "static_routes_unlabeled.jsonl")[pair_id]
    write_json(out / "routing.json", route)
    texts = [
        "# Pair inspection",
        "",
        f"Pair ID: `{pair_id}`",
        "",
        "## Module 1",
        "",
        "```json",
        __import__("json").dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Input code",
        "",
    ]
    for s in ["function_a", "function_b"]:
        texts += [f"### {s}", "", "````c", pairs[pair_id][s]["pseudo_code"], "````", ""]
    trail = []
    annotated = {r["stage"]: r for r in annotate_routes(run) if r["pair_id"] == pair_id}
    specs = [dict(stage="fewshot", gate_summary="Module-one route requires model judgment")] + read_json(
        RESOURCES / "stages.json"
    )
    for spec in specs:
        stage = spec["stage"]
        folder = run / "stages" / stage
        if not (folder / "answers.jsonl").exists():
            trail.append(dict(stage=stage, status="NOT_COMPLETED"))
            continue
        answer = index(folder / "answers.jsonl").get(pair_id)
        texts += [f"## {stage}", "", spec["gate_summary"], ""]
        if answer is None:
            detail = dict(stage=stage, status="NOT_CALLED", gate=spec["gate_summary"])
            trace = run / "routing" / (stage + ".jsonl")
            if stage in annotated:
                detail.update(annotated[stage])
            elif trace.exists():
                detail.update(index(trace)[pair_id])
            trail.append(detail)
            texts += [__import__("json").dumps(detail, ensure_ascii=False), ""]
            continue
        target = out / stage
        target.mkdir()
        write_json(target / "input.json", index(folder / "inputs_unlabeled.jsonl")[pair_id])
        write_json(target / "answer.json", answer)
        calls = [r for r in rows(folder / "requests.jsonl") if r["pair_id"] == pair_id]
        write_rows(target / "requests.jsonl", calls)
        for n, call in enumerate(calls, 1):
            for m in call["payload"]["messages"]:
                texts += [f"### Request {n}: {m['role']}", "", "````text", m["content"], "````", ""]
            texts += [
                "### Raw output",
                "",
                "````text",
                call["response"]["choices"][0]["message"]["content"],
                "````",
                "",
            ]
        trail.append(
            dict(
                stage=stage,
                status="CALLED",
                prediction=answer["prediction"],
                reason=answer["reason"],
                calls=len(calls),
            )
        )
    write_json(out / "trail.json", trail)
    (out / "README.md").write_text("\n".join(texts) + "\n")
    return dict(
        pair_id=pair_id,
        stages_called=sum(r["status"] == "CALLED" for r in trail),
        report=str(out / "README.md"),
    )
