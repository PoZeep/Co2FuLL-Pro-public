"""Evaluate case-level search predictions."""

import argparse
from collections import Counter
import json

from co2full.io import rows


def unique_ids(values, field):
    ids = set(values)
    if len(ids) != len(values) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"{field} must contain unique nonempty candidate IDs")
    return ids


def score(cases):
    counts = Counter()
    seen = set()
    for row in cases:
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"Duplicate case: {case_id}")
        seen.add(case_id)
        positive = unique_ids(row["positives"], "positives")
        retrieved = unique_ids(row["retrieved"], "retrieved")
        accepted = unique_ids(row["accepted"], "accepted")
        excluded = unique_ids(row.get("excluded", []), "excluded")
        if not accepted <= retrieved:
            raise ValueError(f"Accepted candidates outside retrieved set in {case_id}")
        positive -= excluded
        retrieved -= excluded
        accepted -= excluded
        counts["tp"] += len(accepted & positive)
        counts["fp"] += len(accepted - positive)
        counts["retrieval_fn"] += len(positive - retrieved)
        counts["verification_fn"] += len((positive & retrieved) - accepted)
    tp, fp = counts["tp"], counts["fp"]
    fn = counts["retrieval_fn"] + counts["verification_fn"]
    return dict(
        cases=len(seen),
        tp=tp,
        fp=fp,
        fn=fn,
        retrieval_fn=counts["retrieval_fn"],
        verification_fn=counts["verification_fn"],
        precision=tp / (tp + fp) if tp + fp else 0,
        recall=tp / (tp + fn) if tp + fn else 0,
        f2=5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", help="JSONL with one record per search case")
    args = parser.parse_args()
    print(json.dumps(score(rows(args.cases)), indent=2))
