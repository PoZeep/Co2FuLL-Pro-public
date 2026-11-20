from co2full.runtime import asset_root as _cf_asset_root

from pathlib import Path

from collections import Counter

import json, math

ROOT = _cf_asset_root()

from scripts.run_fresh_replay_v168 import write, lines, mm

from scripts.run_xm_prefilter_v100 import token_count

from subtasks.ctree_rescue_v139 import content, render


def allowed(prediction, source, structural, frozen):
    a = structural["score"][frozen["selected"]["low_source"]]
    b = structural["score"][frozen["selected"]["high_source"]]
    cause = structural.get("cause") or structural["arms"]["structural_yes_no_then_models"]["cause"]
    return bool(
        prediction == 0
        and source == "model"
        and cause in ["density_weak_compile_disabled", "adaptive_allows_compile"]
        and a is not None
        and b is not None
        and math.isfinite(a)
        and math.isfinite(b)
        and a >= 0
        and b >= 0
    )


def prompt(row, variant, phase):
    method = ("context" if variant == "tree" else "control") + ("_raw_1" if phase == "raw" else "_confirm_1")
    return content(row, method)


class Experiment:
    def evidence(self, potential):
        mode = "train" if self.training else "xm"
        mapping_path = ROOT / "ctree/mapping.json"
        mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else None
        assert mapping or self.training
        names = {x["binary"]: x["key"] for x in mapping["jobs"]} if mapping else {}
        sources = [mapping_path] if mapping else []
        cache = ROOT / "ctree/legacy"
        folders = [cache / x / "ctree/exports" for x in ["hard_ctree", "medium_ctree"]]
        folders.append(ROOT / "ctree/fresh")
        needed = {
            (f["binary"], int(f["address"], 16))
            for row in potential.values()
            for f in [row["function_a"], row["function_b"]]
        }
        records, origins, conflicts = {}, {}, set()
        for folder in folders:
            for path in sorted(folder.glob("*.json")):
                if path.name.startswith("._"):
                    continue
                x = json.loads(path.read_text())
                if x.get("schema_version") != "hexrays_ctree_control_preorder_v1":
                    continue
                binary = Path(x["input_database"]).name.removesuffix(".i64")
                binary = names.get(binary, binary)
                used = False
                for f in x.get("functions", []):
                    key = binary, int(f["function_ea"], 16)
                    if key not in needed or not f.get("ok"):
                        continue
                    used = True
                    value = dict(nodes=f["ctree"]["nodes"], pseudo_code=f["pseudocode"])
                    if key in records and records[key] != value:
                        conflicts.add(key)
                    else:
                        records[key] = value
                        origins[key] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                if used:
                    sources.append(path)
        for key in conflicts:
            records.pop(key, None)
        lookup, coverage, rendered = {}, [], {}
        for n, (p, row) in enumerate(sorted(potential.items())):
            trees, reason = {}, "available"
            for side in ["function_a", "function_b"]:
                f = row[side]
                key = f["binary"], int(f["address"], 16)
                rec = records.get(key)
                if not rec:
                    reason = "raw_tree_unavailable"
                    continue
                if rec["pseudo_code"] != f["pseudo_code"]:
                    reason = "cached_pseudocode_mismatch"
                    continue
                try:
                    if key not in rendered:
                        rendered[key] = render(rec["nodes"])
                    trees[side] = rendered[key]
                except (AssertionError, RecursionError, KeyError):
                    reason = "invalid_recorded_tree"
            candidate = dict(row, trees=trees)
            nt = None
            if reason == "available" and len(trees) == 2:
                nt = max(
                    token_count(self.tok, mm(prompt(candidate, v, s)))
                    for v in ["cold", "tree"]
                    for s in ["raw", "confirm"]
                )
                if nt + 4096 > 131072:
                    reason = "complete_views_exceed_context"
            ok = reason == "available" and len(trees) == 2
            coverage.append(dict(pair_id=p, available=ok, reason=reason, max_prompt_tokens=nt))
            if ok:
                lookup[p] = candidate
            if n % 50 == 0:
                self.status(stage="tree_evidence", done=n, total=len(potential))
        lines(self.run / "evidence/coverage_unlabeled.jsonl", coverage)
        lines(self.run / "evidence/inputs_unlabeled.jsonl", lookup.values())
        lines(
            self.run / "evidence/raw_records_unlabeled.jsonl",
            [dict(binary=k[0], address=k[1], source=origins[k], **x) for k, x in sorted(records.items())],
        )
        write(
            self.run / "evidence/status.json",
            dict(
                potential=len(potential),
                available=len(lookup),
                conflicts=len(conflicts),
                coverage=dict(Counter(x["reason"] for x in coverage)),
            ),
        )
        return lookup, sources
