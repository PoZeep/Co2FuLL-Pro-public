def evidence(f, policy):
    if f["status"] != "ok":
        return False, False
    count, fraction = policy["count"], policy["fraction"]
    same = (f["target_seed"] and f["target_seed_votes"] >= count) or all(
        f[s]["target_unique_best"] and f[s]["target_support"] >= count and f[s]["target_fraction"] >= fraction
        for s in ("a", "b")
    )
    different = (f["both_seeded_elsewhere"] and f["elsewhere_seed_votes"] >= count) or all(
        f[s]["other_unique_best"]
        and f[s]["best_support"] >= count
        and f[s]["best_fraction"] >= fraction
        and f[s]["target_support"] == 0
        for s in ("a", "b")
    )
    if same and different:
        return False, False
    return bool(same), bool(different)
