from co2full.runtime import (
    asset_root as _cf_asset_root,
    tool_path as _cf_tool_path,
    engine_script as _cf_engine_script,
)

import re, struct, subprocess

ROOT = _cf_asset_root()

from scripts.run_fresh_replay_v168 import read, write, lines, sha

CACHE = ROOT / "helpers/observations.jsonl"


def select(inputs, state):
    pairs = []
    images = {}
    targets = set()
    for p, row in sorted(inputs.items()):
        if state[p]["source"] != "model" or state[p]["prediction"] != 0:
            continue
        text = "\n".join(row[s]["pseudo_code"] for s in ["function_a", "function_b"])
        if not re.search(r"\b(?:HIDWORD|LODWORD|double|float)\b|_f64|_pd\b|_sd\b", text):
            continue
        sides = {}
        for s in ["function_a", "function_b"]:
            f = row[s]
            b = f["binary"]
            path = ROOT / "binaries" / b / "image"
            if not path.exists():
                continue
            with path.open("rb") as h:
                header = h.read(64)
            if (
                header[:4] != b"\x7fELF"
                or header[4:6] != b"\x01\x01"
                or struct.unpack_from("<H", header, 18)[0] != 40
            ):
                continue
            addresses = sorted(
                {
                    int(a, 16)
                    for a in re.findall(r"\b(?:j_)?sub_([0-9a-fA-F]+)\s*\(", f["pseudo_code"])
                    if int(a, 16) != int(f["address"], 16)
                }
            )
            if not addresses:
                continue
            if b not in images:
                images[b] = dict(binary=b, image=str(path.relative_to(ROOT)), image_sha256=sha(path))
            sides[s] = dict(binary=b, addresses=[hex(a) for a in addresses])
            targets.update((b, a) for a in addresses)
        if sides:
            pairs.append(dict(pair_id=p, sides=sides))
    return pairs, images, targets


def build(inputs, state, out, audit_source=None):
    out.mkdir(exist_ok=False)
    pairs, images, targets = select(inputs, state)
    cached = {(r["binary"], int(r["address"], 16)): r for r in read(CACHE)}
    records = {}
    missing = {}
    for b, a in sorted(targets):
        old = cached.get((b, a))
        if old is not None:
            assert old["image_sha256"] == images[b]["image_sha256"]
            records[b, a] = old
        else:
            missing.setdefault(b, dict(images[b], callees=[]))["callees"].append(hex(a))
    lines(out / "selected_unlabeled.jsonl", pairs)
    write(out / "new_jobs_unlabeled.json", list(missing.values()))
    if audit_source:
        old = {
            (r["binary"], int(r["address"], 16)): r
            for r in read(audit_source / "observations_unlabeled.jsonl")
        }
        assert old.keys() == targets
        for k in targets:
            assert old[k]["image_sha256"] == images[k[0]]["image_sha256"]
            if k in records:
                assert old[k] == records[k]
            else:
                records[k] = old[k]
    elif missing:
        subprocess.run(
            [
                str(_cf_tool_path("probe_python")),
                str(_cf_engine_script("probe_helpers_v203.py")),
                str(out / "new_jobs_unlabeled.json"),
                str(out / "new_observations_unlabeled.jsonl"),
            ],
            check=True,
            cwd=ROOT,
        )
        for r in read(out / "new_observations_unlabeled.jsonl"):
            k = r["binary"], int(r["address"], 16)
            assert k not in records and k in targets and r["image_sha256"] == images[k[0]]["image_sha256"]
            records[k] = r
    assert records.keys() == targets
    lines(out / "observations_unlabeled.jsonl", [records[k] for k in sorted(records)])
    usable = {}
    for k, r in records.items():
        v = r["validation"]
        if v and v["usable"]:
            assert len(v["valid_matches"]) == 1
            m = v["valid_matches"][0]
            passed = [z for z in v["validations"] if z["passed"] and z["match"] == m]
            assert (
                len(passed) == 1
                and len(passed[0]["tests"]) == 24
                and all(t["passed"] for t in passed[0]["tests"])
            )
            usable[k] = dict(
                primitive=m["primitive"],
                abi=m["abi"],
                initial_cases=m["probes"],
                additional_cases=24,
                register_variations=2,
                image_sha256=r["image_sha256"],
                origin=r["origin"],
            )
    packets = {}
    for row in pairs:
        packet = {}
        for side, f in row["sides"].items():
            hits = [
                dict(address=a, **usable[f["binary"], int(a, 16)])
                for a in f["addresses"]
                if (f["binary"], int(a, 16)) in usable
            ]
            if hits:
                packet[side] = hits
        packets[row["pair_id"]] = dict(
            pair_id=row["pair_id"], available=bool(packet), observed_helpers=packet
        )
    lines(out / "packets_unlabeled.jsonl", packets.values())
    write(
        out / "protocol.json",
        dict(
            mode="reconstruction" if audit_source else "live_target_selection",
            cache_sha256=sha(CACHE),
            images=images,
            pairs=len(pairs),
            targets=len(targets),
            new_helpers=sum(len(j["callees"]) for j in missing.values()),
            available=sum(p["available"] for p in packets.values()),
            labels_read=False,
        ),
    )
    return packets
