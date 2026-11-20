"""Feature extraction and asset preparation pipeline."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import time

from .io import contained, index, lock, read_json, rows, sha, write_json, write_rows
from .runtime import PACKAGE, RESOURCES, activate, require_server

SIDES = ("function_a", "function_b")
EXTRACT_STEPS = ("idb", "ida", "pcode", "ghidra")
FEATURE_STEPS = ("hermes", "csls", "graph", "views")


def copy_checked(source, target, expected=None):
    source, target = Path(source), Path(target)
    digest = sha(source)
    if expected and digest != expected:
        raise ValueError(f"Source SHA-256 mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(target) != digest:
        raise ValueError(f"Copy SHA-256 mismatch: {target}")
    return digest


def validate_candidates(records):
    seen = set()
    for row in records:
        if set(row) != {"pair_id", *SIDES}:
            raise ValueError(
                "Candidates must contain only pair_id/function_a/function_b; labels stay separate"
            )
        # Worker assignment uses the pair ID's hex prefix.
        if not re.fullmatch(r"[0-9a-f]{64}", row["pair_id"]) or row["pair_id"] in seen:
            raise ValueError("pair_id must be a unique lowercase SHA-256 identifier")
        seen.add(row["pair_id"])
        for side in SIDES:
            f = row[side]
            if set(f) != {"binary", "address"} or not re.fullmatch(r"[A-Za-z0-9_-]+", f["binary"]):
                raise ValueError("Endpoint requires a safe opaque binary key and address only")
            if (
                not isinstance(f["address"], str)
                or hex(int(f["address"], 16)) != f["address"]
                or int(f["address"], 16) < 0
            ):
                raise ValueError("Address must be a canonical nonnegative hex function entry")
    if not seen:
        raise ValueError("Empty candidate set")


def initialize(out, from_assets=None, pairs=None, binaries=None, tools=None, examples=None):
    out = Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError("Use a NEW workspace; existing workspaces have resumable stages")
    source = Path(from_assets).expanduser().resolve() if from_assets else None
    if source:
        if pairs or binaries or tools:
            raise ValueError("--from-assets cannot be combined with raw-input arguments")
        from .assets import validate

        validate(source, True)
        examples = examples or source / "few-shot-examples.json"
        candidates = [
            dict(pair_id=r["pair_id"], **{s: {k: r[s][k] for k in ("binary", "address")} for s in SIDES})
            for r in rows(source / "pairs.jsonl")
        ]
        jobs = read_json(source / "upstream/preparation.json")["jobs"]
        images = [
            dict(
                binary=j["key"], path=str(source / "binaries" / j["key"] / "image"), sha256=j["binary_sha256"]
            )
            for j in jobs
        ]
        tool_root = source / "external"
    else:
        if not all((pairs, binaries, tools)):
            raise ValueError("Provide --from-assets OR --pairs/--binaries/--tools")
        candidates = list(rows(pairs))
        images = list(rows(binaries))
        base = Path(binaries).expanduser().resolve().parent
        images = [dict(r, path=str(base / Path(os.path.expandvars(r["path"])).expanduser())) for r in images]
        tool_root = Path(tools).expanduser().resolve()
    if not examples or not Path(examples).expanduser().is_file():
        raise ValueError("Supply the external few-shot examples with --examples")
    validate_candidates(candidates)
    required = {r[s]["binary"] for r in candidates for s in SIDES}
    by = {}
    for r in images:
        if set(r) != {"binary", "path", "sha256"} or r["binary"] in by:
            raise ValueError("Binary manifest: unique binary/path/sha256 fields required")
        by[r["binary"]] = r
    if required != set(by):
        raise ValueError("Binary manifest must exactly cover candidate binaries")
    out.mkdir(parents=True)
    write_json(out / "status.json", dict(state="INITIALIZING"))
    try:
        data = out / "data"
        write_rows(out / "candidates.jsonl", candidates)
        records = []
        for key in sorted(required):
            r = by[key]
            with Path(r["path"]).open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    raise ValueError(f"Not an ELF image: {key}")
            digest = copy_checked(r["path"], data / "binaries" / key / "image", r["sha256"])
            addresses = sorted(
                {r[s]["address"] for r in candidates for s in SIDES if r[s]["binary"] == key},
                key=lambda a: int(a, 16),
            )
            records.append(dict(key=key, binary=key, addresses=addresses, binary_sha256=digest))
        fixed_tools = [
            "gsat-1.0-v22.jar",
            "cviews-v23/com/gsat/tools/ExportCViewsV23.class",
            "fontconfig-v22.properties",
        ]
        fonts = sorted(tool_root.glob("*.ttf"))
        if not fonts:
            raise ValueError("The GSAT tool bundle must include its headless fonts")
        for name in fixed_tools + [p.name for p in fonts]:
            copy_checked(tool_root / name, data / "external" / name)
        font_config = data / "external/fontconfig-v22.properties"
        font_text = font_config.read_text()
        font_text = re.sub(
            r"(?m)^(filename\.[^=]+=)(.+)$", lambda m: m[1] + Path(m[2].strip()).name, font_text
        )
        font_config.write_text(font_text)
        for name in ["routing-policy.json", "csls-protocol.json"]:
            target = "csls/protocol.json" if name == "csls-protocol.json" else name
            copy_checked(RESOURCES / name, data / target)
        copy_checked(Path(examples).expanduser(), data / "few-shot-examples.json")
        for name in ["compile", "context"]:
            (data / "helpers" / name / "exports").mkdir(parents=True)
            write_json(
                data / "helpers" / name / "protocol.json",
                dict(
                    mode="empty_cache_extract_current_targets_at_inference",
                    exporter_sha256=sha(data / "external/cviews-v23/com/gsat/tools/ExportCViewsV23.class"),
                    gsat_sha256=sha(data / "external/gsat-1.0-v22.jar"),
                ),
            )
        write_rows(data / "helpers/observations.jsonl", [])
        write_json(data / "ctree/mapping.json", dict(jobs=records))
        write_json(data / "upstream/preparation.json", dict(jobs=records, mode="new_extraction"))
        sources = {str(p.relative_to(out)): sha(p) for p in sorted(data.rglob("*")) if p.is_file()}
        sources["candidates.jsonl"] = sha(out / "candidates.jsonl")
        write_json(
            out / "workspace.json",
            dict(
                schema="co2full-build-v1",
                jobs=records,
                pairs=len(candidates),
                input_sha256=sources,
                labels_read=False,
                old_evidence_copied=False,
                source_asset_manifest=sha(source / "manifest.jsonl") if source else None,
            ),
        )
        result = dict(
            state="INPUTS_READY",
            workspace=str(out),
            pairs=len(candidates),
            binaries=len(records),
            endpoints=sum(len(j["addresses"]) for j in records),
            next="co2full-extract idb",
        )
        write_json(out / "status.json", result)
        return result
    except BaseException as exc:
        write_json(out / "status.json", dict(state="FAILED", error=repr(exc)))
        raise


def workspace(path, writable=True):
    ws = Path(path).expanduser().resolve()
    spec = read_json(ws / "workspace.json")
    if spec["schema"] != "co2full-build-v1":
        raise ValueError("Unknown build workspace")
    if writable and (ws / "finalized.json").exists():
        raise ValueError("Workspace sealed; initialize a new workspace for changed evidence")
    for name, digest in spec["input_sha256"].items():
        if sha(contained(ws, name)) != digest:
            raise ValueError(f"Workspace input changed: {name}")
    return ws, spec


def fingerprints(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(map(Path, paths))}


def completed(ws, step, key):
    path = ws / "jobs" / step / key / "status.json"
    st = read_json(path)
    if st.get("state") != "COMPLETE":
        raise ValueError(f"Required stage not complete: {path}")
    for output, digest in st["outputs"].items():
        if sha(contained(ws, output)) != digest:
            raise ValueError(f"Completed output changed: {output}")
    for source, digest in st["inputs"].items():
        if sha(source) != digest:
            raise ValueError(f"Completed input changed: {source}")
    return st


def external_run(command, log, env=None, timeout=1800):
    with Path(log).open("w") as stream:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **(env or {})),
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Tool exited {code}; see {log}")


def cfg_file(ws, key):
    return ws / "data/cfg" / (key + ".json")


def output_paths(ws, step, key):
    data = ws / "data"
    if step == "idb":
        return {"idb": ws / "idbs" / key / "image.i64"}
    if step == "ida":
        return dict(
            code=data / "upstream/code" / (key + ".json"),
            cfg=cfg_file(ws, key),
            ctree=data / "ctree/fresh" / (key + ".json"),
            dg=data / "dg" / (key + ".dg"),
        )
    return {step: (ws / "sog" if step == "pcode" else data / "upstream/ghidra") / (key + ".json")}


def java_command(cfg, data, scratch):
    return [
        cfg["java"],
        "-ea",
        "-Xmx8G",
        "-Djava.awt.headless=true",
        "-Dsun.awt.fontconfig=" + str(data / "external/fontconfig-v22.properties"),
        "-Dsun.java2d.fontpath=" + str(data / "external"),
        "-Duser.home=" + str(scratch / "settings"),
        "-XX:ActiveProcessorCount=2",
    ]


def extract_job(ws, job, step, cfg, retry, plan_only, timeout, imported=None):
    key = job["key"]
    folder = ws / "jobs" / step / key
    outputs = output_paths(ws, step, key)
    image = ws / "data/binaries" / key / "image"
    sources = [image, ws / "candidates.jsonl"]
    if step == "idb":
        if imported:
            if imported["binary_sha256"] != job["binary_sha256"]:
                raise ValueError(f"Imported IDB binary identity mismatch: {key}")
            sources.append(Path(imported["path"]))
        else:
            sources += [Path(cfg["ida"]), PACKAGE / "exporters/idb.py"]
    elif step == "ida":
        completed(ws, "idb", key)
        sources += [ws / "idbs" / key / "image.i64", Path(cfg["ida"])] + [
            PACKAGE / "exporters" / (n + ".py") for n in ["bundle", "ctree", "dg"]
        ]
    else:
        completed(ws, "ida", key)
        sources += [cfg_file(ws, key), Path(cfg["java"])] + [
            p for p in (ws / "data/external").rglob("*") if p.is_file()
        ]
        if cfg.get("java_backend"):
            backend = Path(os.path.expandvars(cfg["java_backend"])).expanduser()
            if not backend.is_absolute():
                backend = Path(cfg["config_file"]).parent / backend
            sources += [backend, PACKAGE / "java_runtime.py"]
    inputs = fingerprints(sources)
    options = dict(
        step=step,
        addresses=job["addresses"],
        decompiler_plugins=cfg.get("ida_decompiler_plugins", []),
        ida_pythonpath=cfg.get("ida_pythonpath"),
        imported=imported,
    )
    if (folder / "status.json").exists():
        prior = read_json(folder / "status.json")
        if prior["state"] == "COMPLETE":
            old = completed(ws, step, key)
            if old["inputs"] != inputs or old["options"] != options:
                raise ValueError(f"Changed inputs/tools for {step}/{key}; use a new workspace")
            return dict(binary=key, state="REUSED", coverage=old.get("coverage", []))
        if not retry and not plan_only:
            raise ValueError(f"Failed/interrupted job {step}/{key}; inspect logs and use --retry")
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder / ("attempt-%03d" % (1 + len(list(folder.glob("attempt-*")))))
    attempt.mkdir()
    scratch = attempt / "scratch"
    report = attempt / "export-report.json"
    env = {}
    if step in ["idb", "ida"] and not imported:
        env["TVHEADLESS"] = "1"
        if cfg.get("ida_pythonpath"):
            env["PYTHONPATH"] = cfg["ida_pythonpath"]
        request = dict(
            output=str(outputs.get("idb", "")),
            report=str(report),
            addresses=job["addresses"],
            binary=key,
            idb_key=f"IDBs/{key}/image.i64",
            outputs={n: str(p) for n, p in outputs.items()},
            decompiler_plugins=cfg.get("ida_decompiler_plugins", []),
        )
        write_json(attempt / "request.json", request)
        script = PACKAGE / "exporters" / ("idb.py" if step == "idb" else "bundle.py")
        command = [
            cfg["ida"],
            "-A",
            "-L" + str(attempt / "ida.log"),
            "-S" + str(script),
            "-Oco2full:" + str(attempt / "request.json"),
        ]
        if step == "idb":
            command += ["-c", "-o" + str(outputs["idb"]), str(image)]
        else:
            command += [str(scratch / "image.i64")]
    elif imported:
        command = ["verified-copy", imported["path"], str(outputs["idb"])]
    else:
        wrapper = read_json(cfg_file(ws, key))
        idbkey, functions = next(iter(wrapper.items()))
        if step == "pcode":
            functions = [f for f in functions if f.get("nodes")]
        write_json(attempt / "cfg.json", {idbkey: functions})
        command = java_command(cfg, ws / "data", scratch)
        jar = str(ws / "data/external/gsat-1.0-v22.jar")
        if step == "pcode":
            command += [
                "-XX:+UseCompressedOops",
                "-jar",
                jar,
                "pcode-extractor-v2",
                "-m",
                "elf",
                "-f",
                str(image),
                "-c",
                str(attempt / "cfg.json"),
                "-of",
                "ALL",
                "-v",
                "1",
                "-opt",
                "0",
                "-p",
                str(scratch / "project"),
                "-n",
                key,
                "-o",
                str(outputs[step]),
            ]
        else:
            command += [
                "-cp",
                str(ws / "data/external/cviews-v23") + os.pathsep + jar,
                "com.gsat.tools.ExportCViewsV23",
                str(image),
                str(attempt / "cfg.json"),
                str(outputs[step]),
                str(scratch / "project"),
            ]
    plan = dict(
        binary=key,
        step=step,
        command=command,
        environment_overrides=env,
        inputs=inputs,
        options=options,
        outputs={n: str(p) for n, p in outputs.items()},
        timeout=timeout,
        execute=not plan_only,
    )
    write_json(attempt / "plan.json", plan)
    if plan_only:
        return dict(binary=key, state="PLANNED", plan=str(attempt / "plan.json"))
    tick = time.time()
    st = dict(
        binary=key,
        step=step,
        state="RUNNING",
        inputs=inputs,
        options=options,
        attempt=str(attempt.relative_to(ws)),
    )
    write_json(folder / "status.json", st)
    try:
        for p in outputs.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                archive = attempt / "previous-output" / p.relative_to(ws)
                archive.parent.mkdir(parents=True, exist_ok=True)
                p.rename(archive)
        if imported:
            copy_checked(imported["path"], outputs["idb"], imported["sha256"])
            write_json(
                report, dict(state="complete", origin="verified_existing_IDB", sha256=imported["sha256"])
            )
        else:
            (scratch / "project").mkdir(parents=True)
            (scratch / "settings").mkdir()
            if step == "ida":
                copy_checked(ws / "idbs" / key / "image.i64", scratch / "image.i64")
            if step == "pcode" and not functions:
                write_json(outputs[step], {idbkey: dict(rebase_offset=0)})
                (attempt / "stdout.log").write_text(
                    "[UNAVAILABLE] No nonempty exact-entry CFG; preserved empty SOG container\n"
                )
            else:
                external_run(command, attempt / "stdout.log", env, timeout)
        coverage = extraction_coverage(step, outputs, job, report)
        if fingerprints(sources) != inputs:
            raise ValueError("Input changed while extracting")
        st.update(
            state="COMPLETE",
            outputs={str(p.relative_to(ws)): sha(p) for p in outputs.values()},
            coverage=coverage,
            seconds=time.time() - tick,
        )
    except Exception as exc:
        st.update(state="FAILED", error=repr(exc), seconds=time.time() - tick)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    write_json(attempt / "status.json", st)
    write_json(folder / "status.json", st)
    return {k: v for k, v in st.items() if k not in ["inputs", "options", "outputs"]}


def extraction_coverage(step, outputs, job, report):
    for path in outputs.values():
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Exporter produced no nonempty output: {path}")
    if step == "idb":
        if read_json(report)["state"] != "complete":
            raise ValueError("IDA did not confirm successful database save")
        return []
    if step == "ida":
        codes = read_json(outputs["code"])
        tree = read_json(outputs["ctree"])
        cfg = next(iter(read_json(outputs["cfg"]).values()))
        if (
            set(codes) != set(job["addresses"])
            or {f["start_ea"] for f in cfg} != set(codes)
            or {f["function_ea"] for f in tree["functions"]} != set(codes)
        ):
            raise ValueError("IDA endpoint coverage mismatch")
        if tree["schema_version"] != "hexrays_ctree_control_preorder_v1":
            raise ValueError("Unexpected ctree schema")
        with outputs["dg"].open("rb") as stream:
            graph = pickle.load(stream)
        if not graph.is_directed():
            raise ValueError("DG must be directed")
        coverage = read_json(report)["functions"]
        for row in coverage:
            row["dg"] = (
                "ok"
                if row["address"] in graph and graph.nodes[row["address"]].get("type") == "func"
                else "unavailable"
            )
        return coverage
    values = read_json(outputs[step])
    if step == "pcode":
        if len(values) != 1:
            raise ValueError("Expected one SOG binary container")
        raw = next(iter(values.values()))
        offset = int(raw.get("rebase_offset", 0))
        return [
            dict(
                address=a,
                status="ok"
                if raw.get(hex(int(a, 16) + offset), {}).get("SOG", {}).get("nodes")
                else "unavailable",
                graph_address=hex(int(a, 16) + offset),
                reason=None
                if raw.get(hex(int(a, 16) + offset), {}).get("SOG", {}).get("nodes")
                else "missing_or_empty_sog",
            )
            for a in job["addresses"]
        ]
    if set(values) != set(job["addresses"]):
        raise ValueError("Ghidra exporter did not return every requested address")
    return [
        dict(
            address=a,
            status=values[a].get("status", "unavailable"),
            reason=values[a].get("reason"),
            loaded_address=values[a].get("loaded_address"),
            load_offset=values[a].get("load_offset"),
        )
        for a in job["addresses"]
    ]


def extract(args):
    require_server()
    cfg = activate(args.config)
    ws = Path(args.workspace).expanduser().resolve()
    with lock(ws / "active.lock"):
        ws, spec = workspace(ws)
        jobs = [j for j in spec["jobs"] if args.binary == "all" or j["key"] == args.binary]
        if not jobs or args.jobs < 1 or args.timeout < 1:
            raise ValueError("Unknown binary or invalid jobs/timeout")
        tool = (
            None
            if args.step == "idb" and args.idb_manifest
            else "ida"
            if args.step in ["idb", "ida"]
            else "java"
        )
        if tool and (not cfg.get(tool) or not Path(cfg[tool]).is_file() or not os.access(cfg[tool], os.X_OK)):
            raise ValueError(f"Configure an existing executable '{tool}' in {args.config}; no jobs launched")
        imported = {}
        if args.idb_manifest:
            if args.step != "idb":
                raise ValueError("--idb-manifest applies to idb only")
            base = Path(args.idb_manifest).resolve().parent
            for r in rows(args.idb_manifest):
                if set(r) != {"binary", "path", "sha256", "binary_sha256"} or r["binary"] in imported:
                    raise ValueError("IDB manifest requires unique binary/path/sha256/binary_sha256")
                imported[r["binary"]] = dict(r, path=str(base / Path(r["path"]).expanduser()))
            if not {j["key"] for j in jobs} <= imported.keys():
                raise ValueError("IDB manifest misses selected binaries")
        results = []

        def job(j):
            try:
                return extract_job(
                    ws, j, args.step, cfg, args.retry, args.plan, args.timeout, imported.get(j["key"])
                )
            except Exception as exc:
                return dict(binary=j["key"], state="FAILED", error=repr(exc))

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(job, j) for j in jobs]):
                result = future.result()
                results.append(result)
                print(
                    f"[{args.step.upper()}] {len(results)}/{len(jobs)} {result['binary']} {result['state']}"
                    + (" " + result["error"] if result.get("error") else ""),
                    flush=True,
                )
                write_json(
                    ws / "logs" / (args.step + "-progress.json"),
                    dict(
                        done=len(results), total=len(jobs), counts=dict(Counter(x["state"] for x in results))
                    ),
                )
        report = dict(
            step=args.step,
            state="FAILED"
            if any(x["state"] == "FAILED" for x in results)
            else "PLANNED"
            if args.plan
            else "COMPLETE",
            counts=dict(Counter(x["state"] for x in results)),
            results=sorted(results, key=lambda r: r["binary"]),
        )
        write_json(ws / "logs" / (args.step + "-summary.json"), report)
        write_rows(
            ws / "logs" / (args.step + "-items.jsonl"),
            [dict(binary=r["binary"], **item) for r in results for item in r.get("coverage", [])],
        )
        if report["state"] == "FAILED":
            raise RuntimeError(f"Extraction failed; see {ws}/logs/{args.step}-summary.json")
        return {k: v for k, v in report.items() if k != "results"}


def primary_pairs(ws, spec):
    codes = {}
    for j in spec["jobs"]:
        completed(ws, "ida", j["key"])
        codes[j["key"]] = read_json(ws / "data/upstream/code" / (j["key"] + ".json"))
    missing, result = [], []
    for r in rows(ws / "candidates.jsonl"):
        item = dict(pair_id=r["pair_id"])
        for side in SIDES:
            f = r[side]
            raw = codes[f["binary"]][f["address"]]
            if raw["status"] != "ok" or not raw.get("pseudo_code", "").strip():
                missing.append(dict(pair_id=r["pair_id"], side=side, **f, reason=raw.get("reason")))
            item[side] = dict(f, pseudo_code=raw.get("pseudo_code", ""))
        result.append(item)
    write_rows(ws / "logs/primary-code-unavailable.jsonl", missing)
    if missing:
        raise ValueError(
            f"{len(missing)} candidate sides lack primary pseudocode. No pairs dropped; see logs/primary-code-unavailable.jsonl"
        )
    return result


def features(args):
    require_server()
    cfg = activate(args.config)
    ws = Path(args.workspace).expanduser().resolve()
    with lock(ws / "active.lock"):
        ws, spec = workspace(ws)
        data = ws / "data"
        dependencies = [ws / "candidates.jsonl", PACKAGE / "data_build.py", PACKAGE / "preparation_core.py"]
        prerequisites = {
            "hermes": ["ida", "pcode"],
            "graph": ["ida"],
            "views": ["ida", "ghidra"],
            "csls": [],
        }[args.step]
        for step in prerequisites:
            for j in spec["jobs"]:
                st = completed(ws, step, j["key"])
                dependencies.extend(ws / p for p in st["outputs"])
        if args.step == "csls":
            st = completed(ws, "features", "hermes")
            dependencies.extend(ws / p for p in st["outputs"])
            dependencies.append(data / "csls/protocol.json")
        if args.step == "hermes":
            checkpoint = Path(cfg["hermes_checkpoint"])
            dependencies += [
                checkpoint,
                checkpoint.parent.parent / "config.json",
                Path(cfg["hermes_vocabulary"]),
                PACKAGE / "hermes_encode.py",
                PACKAGE / "hermes_dependencies.py",
                Path(cfg["hermes_python"]),
            ]
            dependencies += list((Path(cfg["hermes_root"]) / "model/core").glob("*.py"))
            dependencies += list((Path(cfg["hermes_root"]) / "preprocess").glob("*.py"))
        elif args.step == "views":
            dependencies.append(PACKAGE / "view_text.py")
        inputs = fingerprints(set(dependencies))
        folder = ws / "jobs/features" / args.step
        if (folder / "status.json").exists():
            prior = read_json(folder / "status.json")
            if prior["state"] == "COMPLETE":
                st = completed(ws, "features", args.step)
                if st["inputs"] != inputs:
                    raise ValueError("Feature inputs/tools changed; use a new workspace")
                return dict(state="REUSED", step=args.step, summary=st["summary"])
            if not args.retry:
                raise ValueError("Previous attempt failed/interrupted; inspect logs then use --retry")
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / ("attempt-%03d" % (1 + len(list(folder.glob("attempt-*")))))
        attempt.mkdir()
        st = dict(state="RUNNING", inputs=inputs, step=args.step, attempt=str(attempt.relative_to(ws)))
        write_json(folder / "status.json", st)
        try:
            pairs = primary_pairs(ws, spec)
            write_rows(data / "pairs.jsonl", pairs)
            if args.step == "hermes":
                machine = dict(cfg, assets=str(data))
                write_json(attempt / "machine.local.json", machine)
                command = [
                    cfg["hermes_python"],
                    "-u",
                    "-m",
                    "co2full.hermes_encode",
                    "--input",
                    str(data / "pairs.jsonl"),
                    "--graphs",
                    str(ws / "sog"),
                    "--binaries",
                    str(data / "binaries"),
                    "--vocabulary",
                    str(Path(cfg["hermes_vocabulary"])),
                    "--limit",
                    str(len(pairs)),
                    "--run-name",
                    str(attempt / "encoding"),
                ]
                write_json(attempt / "plan.json", dict(command=command))
                external_run(
                    command,
                    attempt / "stdout.log",
                    dict(CO2FULL_CONFIG=str(attempt / "machine.local.json")),
                    args.timeout,
                )
                summary = read_json(attempt / "encoding/manifest.json")
                copy_checked(attempt / "encoding/embeddings.jsonl", data / "hermes/embeddings.jsonl")
                copy_checked(attempt / "encoding/manifest.json", data / "hermes/encoding.json")
                copy_checked(attempt / "encoding/input_hashes.json", data / "hermes/input_hashes.json")
                outputs = [
                    data / "hermes" / n for n in ["embeddings.jsonl", "encoding.json", "input_hashes.json"]
                ]
            elif args.step == "csls":
                from .preparation_core import geometry

                vectors = {
                    (r["binary"], r["address"]): r["vector"]
                    for r in rows(data / "hermes/embeddings.jsonl")
                    if r["status"] == "ok"
                }
                populations = Counter(b for b, a in vectors)
                if not vectors or min(len(vectors) - n for n in populations.values()) < 10:
                    raise ValueError(
                        "Frozen fixed10 CSLS requires at least 10 valid cross-binary neighbors per endpoint; use the full corpus, not a one-binary smoke test"
                    )
                protocol = read_json(data / "csls/protocol.json")
                scores, neighbors, densities = geometry(
                    pairs,
                    vectors,
                    protocol["selected"]["k"],
                    protocol["selected"]["alpha"],
                    protocol["neighbor_fraction"],
                )
                write_rows(data / "csls/scores.jsonl", scores["diffused"])
                write_rows(data / "csls/reciprocal_neighbors.jsonl", neighbors)
                write_rows(data / "csls/densities.jsonl", densities["diffused"])
                outputs = [
                    data / "csls" / n
                    for n in ["scores.jsonl", "reciprocal_neighbors.jsonl", "densities.jsonl"]
                ]
                summary = dict(
                    pairs=len(pairs),
                    finite=sum(r["score"] is not None for r in scores["diffused"]),
                    reference_pool="all_unlabeled_candidate_endpoints",
                    parameters=protocol,
                )
            elif args.step == "graph":
                from .preparation_core import summarize, seed_map, measure
                from functools import lru_cache

                @lru_cache(maxsize=24)
                def graph(key):
                    with (data / "dg" / (key + ".dg")).open("rb") as stream:
                        return summarize(pickle.load(stream))

                groups = defaultdict(list)
                for r in pairs:
                    groups[r["function_a"]["binary"], r["function_b"]["binary"]].append(r)
                result = []
                for n, ((a, b), rr) in enumerate(sorted(groups.items()), 1):
                    ga, gb = graph(a), graph(b)
                    seeds = seed_map(ga, gb)
                    for r in rr:
                        result.append(
                            dict(
                                pair_id=r["pair_id"],
                                graph=measure(
                                    ga, gb, r["function_a"]["address"], r["function_b"]["address"], seeds
                                ),
                            )
                        )
                    if n % 250 == 0:
                        print(f"[GRAPH] {n}/{len(groups)} binary pairs", flush=True)
                write_rows(data / "graph/features.jsonl", result)
                outputs = [data / "graph/features.jsonl"]
                summary = dict(
                    pairs=len(result), statuses=dict(Counter(r["graph"]["status"] for r in result))
                )
            else:
                from .view_text import anonymize

                result, cache = {}, {}
                for r in pairs:
                    for side in SIDES:
                        f = r[side]
                        key = f["binary"], f["address"]
                        if key in result:
                            continue
                        if key[0] not in cache:
                            cache[key[0]] = read_json(data / "upstream/ghidra" / (key[0] + ".json"))
                        raw = cache[key[0]][key[1]]
                        code = raw.get("code", "") if raw.get("status") == "ok" else ""
                        result[key] = dict(
                            binary=key[0],
                            address=key[1],
                            original=f["pseudo_code"],
                            ida=anonymize(f["pseudo_code"]),
                            ghidra=anonymize(code),
                            usable=bool(code),
                        )
                write_rows(data / "views.jsonl", [result[k] for k in sorted(result)])
                outputs = [data / "views.jsonl"]
                summary = dict(endpoints=len(result), usable=sum(r["usable"] for r in result.values()))
            if fingerprints(set(dependencies)) != inputs:
                raise ValueError("Feature input changed during computation")
            st.update(
                state="COMPLETE", outputs={str(p.relative_to(ws)): sha(p) for p in outputs}, summary=summary
            )
        except BaseException as exc:
            st.update(state="FAILED", error=repr(exc))
            raise
        finally:
            write_json(folder / "status.json", st)
            write_json(attempt / "status.json", st)
        return dict(state="COMPLETE", step=args.step, summary=summary)


def finalize(path, out):
    ws = Path(path).expanduser().resolve()
    dest = Path(out).expanduser().resolve()
    if dest.exists() or dest == ws or ws in dest.parents:
        raise ValueError("Final assets must be a new directory outside the build workspace")
    with lock(ws / "active.lock"):
        ws, spec = workspace(ws)
        reports = []
        for step in EXTRACT_STEPS:
            for j in spec["jobs"]:
                reports.append(completed(ws, step, j["key"]))
        for step in FEATURE_STEPS:
            reports.append(completed(ws, "features", step))
        pairs = primary_pairs(ws, spec)
        data = ws / "data"
        if list(rows(data / "pairs.jsonl")) != pairs:
            raise ValueError("Primary code changed after feature generation")
        scores, graphs = index(data / "csls/scores.jsonl"), index(data / "graph/features.jsonl")
        if set(scores) != {r["pair_id"] for r in pairs} or set(scores) != set(graphs):
            raise ValueError("Pair membership changed during feature preparation")
        write_rows(
            data / "structure.jsonl",
            [
                dict(pair_id=r["pair_id"], score=scores[r["pair_id"]], graph=graphs[r["pair_id"]]["graph"])
                for r in pairs
            ],
        )
        prep = dict(
            mode="fresh_server_extraction",
            jobs=[
                dict(
                    j,
                    idb_sha256=sha(ws / "idbs" / j["key"] / "image.i64"),
                    cfg_sha256=sha(cfg_file(ws, j["key"])),
                )
                for j in spec["jobs"]
            ],
        )
        shutil.copytree(data, dest)
        write_json(dest / "upstream/preparation.json", prep)
        write_json(
            dest / "upstream/build-provenance.json",
            dict(
                workspace_schema=spec["schema"],
                candidates_sha256=sha(ws / "candidates.jsonl"),
                build_source_sha256=sha(Path(__file__)),
                stage_reports=reports,
                old_evidence_copied=False,
                labels_read=False,
            ),
        )
        records = [
            dict(path=str(p.relative_to(dest)), bytes=p.stat().st_size, sha256=sha(p), status="PREPARED")
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        ]
        write_rows(dest / "manifest.jsonl", records)
        try:
            from .assets import validate

            result = validate(dest, True)
            result.update(
                state="READY", mode="fresh_server_extraction", old_evidence_copied=False, assets=str(dest)
            )
            write_json(dest / "status.json", result)
            write_json(
                ws / "finalized.json", dict(assets=str(dest), manifest_sha256=sha(dest / "manifest.jsonl"))
            )
            return result
        except BaseException as exc:
            write_json(dest / "status.json", dict(state="FAILED", error=repr(exc)))
            raise


def status(path):
    ws, spec = workspace(path, writable=False)
    summaries, items = {}, []
    for step in EXTRACT_STEPS:
        counts = Counter()
        for j in spec["jobs"]:
            file = ws / "jobs" / step / j["key"] / "status.json"
            st = read_json(file) if file.exists() else dict(state="NOT_RUN")
            counts[st["state"]] += 1
            for r in st.get("coverage", []):
                items.append(dict(step=step, binary=j["key"], **r))
        summaries[step] = dict(counts)
    for step in FEATURE_STEPS:
        file = ws / "jobs/features" / step / "status.json"
        st = read_json(file) if file.exists() else dict(state="NOT_RUN")
        summaries[step] = {k: v for k, v in st.items() if k in ["state", "summary", "error"]}
    write_rows(ws / "logs/coverage.jsonl", items)
    coverage_counts = {}
    for field in ["code", "cfg", "dg", "ctree"]:
        coverage_counts[field] = dict(Counter(r[field] for r in items if r["step"] == "ida"))
    for step in ["pcode", "ghidra"]:
        coverage_counts[step] = dict(Counter(r["status"] for r in items if r["step"] == step))
    result = dict(
        workspace=str(ws),
        pairs=spec["pairs"],
        binaries=len(spec["jobs"]),
        stages=summaries,
        coverage_counts=coverage_counts,
        sealed=(ws / "finalized.json").exists(),
        coverage=str(ws / "logs/coverage.jsonl"),
    )
    write_json(ws / "logs/summary.json", result)
    return result
