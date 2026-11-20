"""Asset validation, packaging, and extraction utilities."""

from pathlib import Path
from collections import Counter
import math
import os
import shutil
import tarfile
import time
from .io import contained, event, index, read_json, rows, sha, write_json, write_rows
from .runtime import RESOURCES


def validate(root, full=False):
    root = Path(root).resolve()
    records = list(rows(root / "manifest.jsonl"))
    paths = [x["path"] for x in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manifest entries")
    for r in records:
        p = contained(root, r["path"])
        if not p.is_file() or p.stat().st_size != r["bytes"] or (full and sha(p) != r["sha256"]):
            raise ValueError(f"Asset invalid or missing: {r['path']}")
    pairs = index(root / "pairs.jsonl")
    endpoints = {}
    for p, r in pairs.items():
        if set(r) != {"pair_id", "function_a", "function_b"}:
            raise ValueError(f"Unexpected inference fields: {p}")
        for s in ["function_a", "function_b"]:
            f = r[s]
            if set(f) != {"binary", "address", "pseudo_code"} or not f["pseudo_code"].strip():
                raise ValueError(f"Missing/invalid primary code: {p}/{s}")
            if hex(int(f["address"], 16)) != f["address"]:
                raise ValueError(f"Noncanonical address: {p}/{s}")
            key = (f["binary"], f["address"])
            if key in endpoints and endpoints[key] != f["pseudo_code"]:
                raise ValueError(f"Conflicting pseudocode: {key}")
            endpoints[key] = f["pseudo_code"]
    structure = index(root / "structure.jsonl")
    scores = index(root / "csls/scores.jsonl")
    if pairs.keys() != structure.keys() or pairs.keys() != scores.keys():
        raise ValueError("Complete candidate membership not preserved")
    finite = 0
    for p, s in scores.items():
        vs = [s[k] for k in ["cosine", "fixed10", "score"]]
        if not (
            all(v is None for v in vs) or all(isinstance(v, (int, float)) and math.isfinite(v) for v in vs)
        ):
            raise ValueError(f"Malformed CSLS availability: {p}")
        finite += vs[0] is not None
        if any(structure[p]["score"][k] != s[k] for k in ["cosine", "fixed10", "score"]):
            raise ValueError(f"Stale CSLS in structure: {p}")
        if set(structure[p]) != {"pair_id", "score", "graph"}:
            raise ValueError("Historical model fields must not enter structure")
    vectors = list(rows(root / "hermes/embeddings.jsonl"))
    keys = [(r["binary"], r["address"]) for r in vectors]
    if len(keys) != len(set(keys)) or set(keys) != set(endpoints):
        raise ValueError("Embedding endpoint coverage mismatch")
    valid = 0
    for r in vectors:
        if r["status"] == "ok":
            v = r.get("vector", [])
            if len(v) != 384 or not all(math.isfinite(x) for x in v) or sum(x * x for x in v) == 0:
                raise ValueError("Invalid successful embedding")
            valid += 1
        elif r["status"] != "unavailable":
            raise ValueError("Unknown embedding status")
    views = list(rows(root / "views.jsonl"))
    vk = [(r["binary"], r["address"]) for r in views]
    if len(vk) != len(set(vk)) or set(vk) != set(endpoints):
        raise ValueError("View endpoint coverage mismatch")
    if any(r["original"] != endpoints[(r["binary"], r["address"])] for r in views):
        raise ValueError("View pseudocode does not match pair code")
    return dict(
        state="VERIFIED",
        pairs=len(pairs),
        endpoints=len(endpoints),
        binaries=len({k[0] for k in endpoints}),
        embeddings_ok=valid,
        embeddings_unavailable=len(vectors) - valid,
        csls_finite=finite,
        csls_unavailable=len(pairs) - finite,
        ghidra_usable=sum(bool(r.get("usable") and r.get("ghidra")) for r in views),
        files=len(records),
        full_hash_check=full,
        manifest_sha256=sha(root / "manifest.jsonl"),
    )


def pack(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    validate(root, True)
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Archive must be a new file outside the asset directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz", compresslevel=1) as archive:
        for r in rows(root / "manifest.jsonl"):
            archive.add(contained(root, r["path"]), arcname=r["path"], recursive=False)
        archive.add(root / "manifest.jsonl", arcname="manifest.jsonl")
    result = dict(path=str(output), bytes=output.stat().st_size, sha256=sha(output))
    write_json(str(output) + ".sha256.json", result)
    return result


def unpack(archive, dest):
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    seen = set()
    try:
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle:
                if member.name in seen or not member.isfile():
                    raise ValueError("Only unique regular files are permitted in asset bundles")
                seen.add(member.name)
                target = contained(dest, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out)
        return validate(dest, True)
    except BaseException as exc:
        write_json(dest / "import-error.json", dict(error=repr(exc)))
        raise
