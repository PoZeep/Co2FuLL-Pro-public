"""Execution pipeline for CSS routing, Pre-Verification, and Post-Verification."""

from pathlib import Path
from collections import Counter
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from .io import event, index, lock, read_json, rows, sha, write_json, write_rows
from .runtime import activate, asset_root, config, ENGINE, PACKAGE, RESOURCES, require_server
from .doctor import engine_check
from .assets import validate
from . import services
from .ablation import verification_gates


class ModuleBoundary(Exception):
    pass


def source_fingerprints():
    files = list(ENGINE.rglob("*.py")) + [
        PACKAGE / n
        for n in [
            "pipeline.py",
            "runtime.py",
            "services.py",
            "ablation.py",
            "single_gpu_broker.py",
            "single_gpu_native_worker.py",
        ]
    ]
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(files)}


def verify_protocol(run):
    pr = read_json(run / "protocol.json")
    for rel, digest in pr["source_sha256"].items():
        if sha(PACKAGE / rel) != digest:
            raise RuntimeError(f"Code changed during run: {rel}")
    cfg = config(run / "machine.json")
    if sha(Path(cfg["assets"]) / "manifest.jsonl") != pr["asset_manifest_sha256"]:
        raise RuntimeError("Asset manifest changed during run")
    for r in rows(Path(cfg["assets"]) / "manifest.jsonl"):
        p = Path(cfg["assets"]) / r["path"]
        if not p.is_file() or p.stat().st_size != r["bytes"]:
            raise RuntimeError(f"Asset missing/changed: {r['path']}")
    return pr, cfg


def seal(run, module, files, **details):
    value = dict(module=module, files={str(p.relative_to(run)): sha(p) for p in files}, **details)
    write_json(run / f"module{module}.json", value)
    return value


def check_seal(run, module):
    for rel, digest in read_json(run / f"module{module}.json")["files"].items():
        if sha(run / rel) != digest:
            raise RuntimeError(f"Module output changed: {rel}")


def route(path, run):
    cfg = activate(path)
    engine_check()
    report = validate(cfg["assets"], True)
    variant = cfg.get("variant", "full")
    if variant not in ("full", "no-css"):
        raise ValueError(f"Unknown variant: {variant}")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "machine.json", cfg)
    write_json(
        run / "protocol.json",
        dict(
            project="Co2FuLL-Pro",
            version="1.0.0",
            variant=variant,
            mode="fresh_inference",
            created=time.time(),
            asset_manifest_sha256=report["manifest_sha256"],
            source_sha256=source_fingerprints(),
            input_sha256=sha(Path(cfg["assets"]) / "pairs.jsonl"),
            labels_read=False,
            modules=["structural_routing", "fewshot", "structured_postverification"],
            stages=["fewshot"] + [s["stage"] for s in read_json(RESOURCES / "stages.json")],
            inference="Fixed sorted/shuffled ordered batches of 32, BF16 Qwen2.5-7B, unchanged prompts and recovery protocol",
        ),
    )
    from subtasks.structural_triage_v106 import route as decide_route

    if variant == "no-css":
        from .ablation import all_model_route as decide_route
    data = index(Path(cfg["assets"]) / "pairs.jsonl")
    structure = {
        p: {k: r[k] for k in ["score", "graph"]}
        for p, r in index(Path(cfg["assets"]) / "structure.jsonl").items()
    }
    policy = read_json(Path(cfg["assets"]) / "routing-policy.json")
    result = [
        dict(
            pair_id=p,
            **decide_route(structure[p]["score"], structure[p]["graph"], policy, direct_no=True),
            **structure[p],
        )
        for p in sorted(data)
    ]
    write_rows(run / "static_routes_unlabeled.jsonl", result)
    counts = dict(Counter(r["action"] for r in result))
    seal(
        run,
        1,
        [run / "static_routes_unlabeled.jsonl", run / "machine.json", run / "protocol.json"],
        pairs=len(data),
        actions=counts,
    )
    write_json(
        run / "status.json", dict(state="MODULE1_COMPLETE", next_command="co2full-fewshot", actions=counts)
    )
    return dict(state="MODULE1_COMPLETE", pairs=len(data), actions=counts, run=str(run))


def make_pipeline(cfg, run, module, reference=None):
    activate(cfg["config_file"])
    from scripts.run_stable_pipeline_v203 import Pipeline
    from scripts.run_fresh_replay_v168 import Replay
    from scripts.audit_fresh_replay_v168 import Auditor
    from scripts.offline_replay_v178 import OfflineReplay
    from scripts.run_xm_prefilter_v100 import tokenizer
    import baseline_fewshot as base

    class LoggedPipeline(Pipeline):
        def status(self, state="running", **kw):
            value = dict(
                state=state,
                module=module,
                pid=os.getpid(),
                calls=self.calls,
                elapsed_seconds=time.time() - self.start,
                stage_counts=self.stage_counts,
                **kw,
            )
            write_json(self.run / "status.json", value)
            now = time.monotonic()
            if now - getattr(self, "last_print", 0) > 15 or kw.get("phase") == "complete":
                print(json.dumps({k: v for k, v in value.items() if k != "stage_counts"}), flush=True)
                self.last_print = now

        def backend(self, stage, candidates, make_messages):
            if module == 2 and stage == "compile":
                raise ModuleBoundary()
            if not self.auditing and stage != "fewshot" and (self.run / "stages" / stage).exists():
                raise RuntimeError(
                    f"Partial stage {stage} already exists. Preserve this run and create a new run."
                )
            event(
                self.run / "events.jsonl",
                kind="stage_enter",
                stage=stage,
                candidates=len(candidates),
                mode="archive_audit"
                if self.auditing
                else "own_fewshot_reconstruction"
                if module == 3 and stage == "fewshot"
                else "fresh_inference",
            )
            specs = {r["stage"]: r for r in read_json(RESOURCES / "stages.json")}
            gate = specs.get(stage, {}).get("gate_summary", "Module-one model route")
            coverage_files = {
                "helper132": "helper_evidence/coverage.json",
                "control134": "dual_coverage.jsonl",
                "tree_raw182": "evidence/coverage_unlabeled.jsonl",
                "data184": "facts184_coverage_unlabeled.jsonl",
                "role186": "role186_coverage_unlabeled.jsonl",
                "plain188": "address188_coverage_unlabeled.jsonl",
                "reconcile191": "coverage191_unlabeled.jsonl",
                "idiom194": "coverage194_unlabeled.jsonl",
                "compact195": "coverage195_unlabeled.jsonl",
                "plain199": "coverage199_unlabeled.jsonl",
            }
            coverage_path = self.run / coverage_files.get(stage, "nonexistent-coverage")
            coverage = index(coverage_path) if coverage_path.is_file() else {}
            details = []
            for pid in sorted(self.inputs):
                detail = dict(pair_id=pid, stage=stage, called=pid in candidates, gate=gate)
                if pid in candidates:
                    detail["reason"] = "gate_and_evidence_checks_passed"
                else:
                    cov = coverage.get(pid)
                    detail["reason"] = "gate_not_satisfied"
                    if cov and not cov.get("available"):
                        detail["evidence_check"] = cov
                        detail["reason"] = cov.get("reason") or (
                            "context_budget_exceeded"
                            if cov.get("max_prompt_tokens", cov.get("max_tokens")) is not None
                            else "required_evidence_unavailable"
                        )
                    if pid in getattr(self, "previous_snapshot", {}):
                        detail["retained_at_last_snapshot"] = self.previous_snapshot[pid]
                    elif pid in self.actual_last:
                        detail["last_model_prediction"] = self.actual_last[pid]["answer"]["prediction"]
                details.append(detail)
            write_rows(self.run / "routing" / (stage + ".jsonl"), details)
            if self.auditing or (module == 3 and stage == "fewshot"):
                aa = Auditor.execute(self, stage, candidates, make_messages)
                self.calls += len(candidates) + self.stage_counts[stage].get("repairs", 0)
            else:
                aa = OfflineReplay.execute(self, stage, candidates, make_messages)
            event(
                self.run / "events.jsonl",
                kind="stage_complete",
                stage=stage,
                candidates=len(candidates),
                yes=sum(x["prediction"] for x in aa.values()),
            )
            return aa

        def snapshot(self, name, state):
            before = getattr(self, "previous_snapshot", {})
            changes = [
                dict(pair_id=p, before=before.get(p), after=x)
                for p, x in sorted(state.items())
                if before.get(p) != x
            ]
            write_rows(self.run / "transitions" / (name + ".jsonl"), changes)
            self.previous_snapshot = copy.deepcopy(state)
            super().snapshot(name, state)

    p = LoggedPipeline.__new__(LoggedPipeline)
    p.run = run
    p.source = Path(reference).resolve() if reference else run
    p.playback = False
    p.auditing = bool(reference)
    p.training = False
    p.building_base = False
    p.repeat = 1
    p.start = time.time()
    p.calls = 0
    p.stage_counts = {}
    p.snapshots = {}
    p.latest = {}
    p.actual_last = {}
    p.checked = 0
    root = Path(cfg["assets"])
    p.inputs = index(root / "pairs.jsonl")
    p.structure = {
        pid: {k: r[k] for k in ["score", "graph"]} for pid, r in index(root / "structure.jsonl").items()
    }
    p.frozen = read_json(root / "routing-policy.json")
    p.tok = tokenizer()
    p.examples = base.load_few_shot_messages(root / "few-shot-examples.json", "pseudo_code")
    p.views = {(r["binary"], r["address"]): r for r in rows(root / "views.jsonl")}
    p.tree_records = None
    return p


def native_batch_audit(run, identities):
    batches = {p: {} for p in [8171, 8172, 8173]}
    for file in sorted((run / "stages").glob("*/requests.jsonl")):
        for r in rows(file):
            b = r.get("offline_batch")
            if not b:
                continue
            port = r["transport_endpoint"]
            if b["worker_pid"] != identities[str(port)]["pid"]:
                raise RuntimeError("Native service restarted during run")
            entry = batches[port].setdefault(
                b["number"], dict(stage=file.parent.name, ordered=b["ordered_pairs"], ids=[])
            )
            if entry["stage"] != file.parent.name or entry["ordered"] != b["ordered_pairs"]:
                raise RuntimeError("Native batch changed")
            entry["ids"].append(r["pair_id"])
    for port, bb in batches.items():
        if sorted(bb) != list(range(1, len(bb) + 1)):
            raise RuntimeError(
                f"Foreign or missing native batch on {port}; cold-start native services before a fresh run"
            )
        if any(r["ids"] != r["ordered"] for r in bb.values()):
            raise RuntimeError("Native batch order mismatch")
    return dict(state="PASSED", batches={str(p): len(v) for p, v in batches.items()})


def run_module(module, run):
    require_server()
    run = Path(run).resolve()
    with lock(run / "active.lock"):
        pr, cfg = verify_protocol(run)
        activate(run / "machine.json")
        cfg = config()
        check_seal(run, 1)
        if (run / f"module{module}.json").exists():
            check_seal(run, module)
            return dict(state="ALREADY_COMPLETE", module=module)
        if module == 3:
            folder = Path(cfg["assets"])
            inputs = {
                row[side]["binary"]
                for row in rows(folder / "pairs.jsonl")
                for side in ("function_a", "function_b")
            }
            missing = sorted(key for key in inputs if not (folder / "binaries" / key / "image").is_file())
            if missing:
                raise ValueError(
                    f"Post-verification needs {len(missing)} missing ELF images. Use an asset directory with the matched ELF images before starting a full run."
                )
        identities = services.verify(cfg)
        if module == 2:
            if (run / "stages/fewshot").exists():
                raise RuntimeError("Partial few-shot run exists; use a new run")
            write_json(run / "services.json", identities)
        else:
            check_seal(run, 2)
            if identities != read_json(run / "services.json"):
                raise RuntimeError("Services changed since few-shot; use a new complete run")
            native_batch_audit(run, identities)
        try:
            p = make_pipeline(cfg, run, module)
            try:
                with verification_gates(cfg.get("variant", "full")):
                    p.pipeline_selected()
            except ModuleBoundary:
                if module != 2:
                    raise
            if services.verify(cfg) != identities:
                raise RuntimeError("Service identity changed")
            batch_report = native_batch_audit(run, identities)
            if module == 2:
                files = sorted(x for x in (run / "stages/fewshot").iterdir() if x.is_file()) + [
                    run / "services.json"
                ]
                result = seal(run, 2, files, calls=p.calls, stage_counts=p.stage_counts, batches=batch_report)
                state = "MODULE2_COMPLETE"
            else:
                final = run / "snapshots/final_helper_pruned016.jsonl"
                if set(index(final)) != set(p.inputs):
                    raise RuntimeError("Final candidate membership differs")
                write_rows(run / "predictions.jsonl", rows(final))
                write_json(
                    run / "decisions_fixed_before_evaluation.json",
                    dict(
                        time=time.time(),
                        protocol_sha256=sha(run / "protocol.json"),
                        snapshots={k: sha(v) for k, v in p.snapshots.items()},
                        predictions_sha256=sha(run / "predictions.jsonl"),
                    ),
                )
                files = [run / "predictions.jsonl", run / "decisions_fixed_before_evaluation.json"] + list(
                    p.snapshots.values()
                )
                result = seal(run, 3, files, calls=p.calls, stage_counts=p.stage_counts, batches=batch_report)
                state = "PREDICTIONS_SEALED"
            write_json(
                run / "status.json",
                dict(
                    state=state,
                    module=module,
                    calls=p.calls,
                    stage_counts=p.stage_counts,
                    elapsed_seconds=time.time() - p.start,
                ),
            )
            return dict(state=state, module=module, calls=p.calls, run=str(run))
        except BaseException as exc:
            write_json(
                run / "status.json",
                dict(state="FAILED", module=module, error=repr(exc), traceback=traceback.format_exc()),
            )
            raise


def detached(args, run, logname, cfg):
    run = Path(run)
    log = run / logname
    with log.open("x") as handle:
        proc = subprocess.Popen(
            [cfg["python"], "-u", "-m", "co2full.pipeline", *args],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        run / (logname + ".process.json"), dict(identity=services.identity(proc.pid), args=args, log=str(log))
    )
    return dict(state="RUNNING", pid=proc.pid, log=str(log), status=str(run / "status.json"))


def command(module, run, path=None, detach=False):
    run = Path(run).expanduser().resolve()
    if module == 1:
        return route(path, run)
    cfg = activate(run / "machine.json")
    if detach:
        return detached(["module", str(module), str(run)], run, f"module{module}.log", cfg)
    return run_module(module, run)


def run_audit(path, reference, out):
    require_server()
    cfg = activate(path)
    out = Path(out).resolve()
    reference = Path(reference).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        p = make_pipeline(cfg, out, 3, reference)
        with verification_gates(cfg.get("variant", "full")):
            p.pipeline_selected()
        expected = sum(
            len(list(rows(f))) for f in (reference / "stages").glob("*/requests_frozen.jsonl")
        ) + sum(len(list(rows(f))) for f in (reference / "stages").glob("*/repair_requests_frozen.jsonl"))
        if p.checked != expected:
            raise RuntimeError(f"Request count mismatch {p.checked}/{expected}")
        result = dict(
            state="PASSED",
            mode="archive_reconstruction_no_inference",
            requests_checked=p.checked,
            snapshots_checked=len(p.snapshots),
            source_sha256=source_fingerprints(),
            reference=str(reference),
            asset_manifest_sha256=sha(Path(cfg["assets"]) / "manifest.jsonl"),
        )
        write_json(out / "audit.json", result)
        write_json(out / "status.json", result)
        return result
    except BaseException as exc:
        write_json(
            out / "status.json", dict(state="FAILED", error=repr(exc), traceback=traceback.format_exc())
        )
        raise


def audit(path, reference, out, detach=False):
    cfg = activate(path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if detach:
        return detached(
            ["audit", cfg["config_file"], str(Path(reference).resolve()), str(out)], out, "audit.log", cfg
        )
    return run_audit(path, reference, out)


if __name__ == "__main__":
    if sys.argv[1] == "module":
        value = run_module(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "audit":
        value = run_audit(*sys.argv[2:5])
    else:
        raise ValueError("Unknown worker action")
    print(json.dumps(value, ensure_ascii=False), flush=True)
