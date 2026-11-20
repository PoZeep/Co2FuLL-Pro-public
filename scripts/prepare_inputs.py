"""Convert a Co²FuLL top-K candidate CSV and binary directory into input manifests."""

import argparse
import csv
import hashlib
from pathlib import Path
import tempfile

from co2full.data_build import validate_candidates
from co2full.io import sha, write_rows


def prepare(csv_path, binary_dir, out, write_labels=False):
    endpoints = []
    labels = []
    names = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"bin_name_1", "fva_1", "bin_name_2", "fva_2"}
        if write_labels:
            required.add("label")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing top-K CSV columns: {sorted(missing)}")
        for row in reader:
            pair = []
            for i in (1, 2):
                name = row[f"bin_name_{i}"]
                if not name or Path(name).name != name:
                    raise ValueError(f"Expected a binary filename: {name!r}")
                address = hex(int(row[f"fva_{i}"], 16))
                pair.append((name, address))
                names.add(name)
            endpoints.append(pair)
            if write_labels:
                if row["label"] not in ("0", "1"):
                    raise ValueError("Expected binary 0/1 evaluation labels")
                labels.append(int(row["label"]))
    if not endpoints:
        raise ValueError("Empty candidate table")
    binary_ids = {name: f"x{i:04d}" for i, name in enumerate(sorted(names))}
    if not binary_dir.is_dir():
        raise NotADirectoryError(binary_dir)
    manifest = {}
    for p in binary_dir.rglob("*"):
        if p.name not in binary_ids or not p.is_file():
            continue
        key = binary_ids[p.name]
        digest = sha(p)
        if key in manifest and manifest[key]["sha256"] != digest:
            raise ValueError(f"Multiple different binaries named {p.name}")
        manifest[key] = dict(binary=key, path=str(p.resolve()), sha256=digest)
    if len(manifest) != len(names):
        missing = [n for n in sorted(names) if binary_ids[n] not in manifest]
        raise FileNotFoundError(f"Missing {len(missing)} binaries (first: {missing[0]})")
    candidates = []
    for a, b in endpoints:
        pair_id = hashlib.sha256(f"{a[0]}@{a[1]}#{b[0]}@{b[1]}".encode()).hexdigest()
        candidates.append(
            dict(
                pair_id=pair_id,
                function_a=dict(binary=binary_ids[a[0]], address=a[1]),
                function_b=dict(binary=binary_ids[b[0]], address=b[1]),
            )
        )
    validate_candidates(candidates)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        prepared = Path(temp) / "inputs"
        write_rows(prepared / "candidates.jsonl", candidates)
        write_rows(prepared / "binaries.jsonl", [manifest[k] for k in sorted(manifest)])
        if write_labels:
            write_rows(
                prepared / "evaluation" / "labels.jsonl",
                (dict(pair_id=row["pair_id"], label=label) for row, label in zip(candidates, labels)),
            )
        prepared.rename(out)
    return len(candidates), len(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True, help="Root of the extracted ELF files")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-labels", action="store_true", help="Export evaluation labels separately")
    args = parser.parse_args()
    n, m = prepare(args.candidates, args.binaries, args.out, args.write_labels)
    print(f"Wrote {n} candidate pairs and {m} binary file references to {args.out}")
