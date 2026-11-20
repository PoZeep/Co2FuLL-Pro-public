"""CSLS similarity calibration and call-graph feature computation."""

from collections import Counter, defaultdict


def geometry(rows, vectors, k, alpha, fraction):
    import numpy as np

    endpoints = {(r[s]["binary"], r[s]["address"]) for r in rows for s in ["function_a", "function_b"]}
    keys = sorted(endpoints & set(vectors))
    pos = {key: i for i, key in enumerate(keys)}
    x = np.asarray([vectors[key] for key in keys], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1)[:, None]
    binary = np.asarray([key[0] for key in keys])
    sim = x @ x.T
    nearest = []
    for i in range(len(keys)):
        valid = np.flatnonzero(binary != binary[i])
        order = np.lexsort((valid, -sim[i, valid]))
        nearest.append(valid[order[:k]].tolist())
    sets = list(map(set, nearest))
    mean = x.copy()
    neighbors = []
    for i, key in enumerate(keys):
        nn = [j for j in nearest[i] if i in sets[j]]
        if nn:
            w = np.maximum(sim[i, nn], 0.0) ** 2
            if w.sum() > 0:
                mean[i] = w @ x[nn] / w.sum()
        neighbors.append(
            dict(
                binary=key[0],
                address=key[1],
                reciprocal=[dict(binary=keys[j][0], address=keys[j][1], cosine=float(sim[i, j])) for j in nn],
            )
        )
    z = (1 - alpha) * x + alpha * mean
    z /= np.linalg.norm(z, axis=1)[:, None]
    scores = {}
    dens = {}
    for name, ss in [("identity", sim), ("diffused", z @ z.T)]:
        ds = {}
        for i, key in enumerate(keys):
            v = ss[i, binary != key[0]]
            n = min(len(v), max(1, int(np.ceil(fraction * len(v)))))
            ds[key] = dict(
                fixed=float(np.partition(v, -10)[-10:].mean()),
                adaptive=float(np.partition(v, -n)[-n:].mean()),
                neighbors=n,
                population=len(v),
            )
        out = []
        for r in rows:
            a, b = [(r[s]["binary"], r[s]["address"]) for s in ["function_a", "function_b"]]
            d = dict(pair_id=r["pair_id"], cosine=None, fixed10=None, score=None)
            if a in pos and b in pos:
                cosine = float(ss[pos[a], pos[b]])
                d.update(
                    cosine=cosine,
                    fixed10=2 * cosine - ds[a]["fixed"] - ds[b]["fixed"],
                    score=2 * cosine - ds[a]["adaptive"] - ds[b]["adaptive"],
                )
            out.append(d)
        scores[name] = out
        dens[name] = [dict(binary=key[0], address=key[1], **ds[key]) for key in keys]
    return scores, neighbors, dens


def summarize(graph):
    nodes = {
        n
        for n, d in graph.nodes(data=True)
        if d.get("type") == "func" and "FUNC_THUNK" not in d.get("tags", [])
    }
    outgoing = {n: set(graph.successors(n)) & nodes - {n} for n in nodes}
    incoming = {n: set(graph.predecessors(n)) & nodes - {n} for n in nodes}
    owners = defaultdict(set)
    for n, d in graph.nodes(data=True):
        value = str(d.get("value", ""))
        if (
            d.get("type") == "str"
            and 6 <= len(value) <= 300
            and not value.startswith(("/home/", "/public/", "/Users/", "/tmp/"))
        ):
            owners[value].update(set(graph.predecessors(n)) & nodes)
    unique = {s: next(iter(v)) for s, v in owners.items() if len(v) == 1}
    return dict(nodes=nodes, outgoing=outgoing, incoming=incoming, unique=unique)


def seed_map(a, b):
    votes = Counter((a["unique"][s], b["unique"][s]) for s in a["unique"].keys() & b["unique"].keys())
    forward, reverse = defaultdict(Counter), defaultdict(Counter)
    for (u, v), count in votes.items():
        forward[u][v] = count
        reverse[v][u] = count

    def best(counter):
        top = max(counter.values())
        winners = [n for n, c in counter.items() if c == top]
        return winners[0] if len(winners) == 1 and top >= 1 else None

    mapping = {
        u: v for u, counter in forward.items() if (v := best(counter)) is not None and best(reverse[v]) == u
    }
    return mapping, {v: u for u, v in mapping.items()}, votes


def neighborhood_scores(a, b, node, mapping):
    """Count votes from the initial anchors."""
    witnesses = defaultdict(set)
    possible = set()
    for direction, reverse in (("outgoing", "incoming"), ("incoming", "outgoing")):
        for anchor in a[direction][node]:
            mate = mapping.get(anchor)
            if mate is None:
                continue
            # High-degree anchors provide weak evidence for a specific match.
            if max(len(a[reverse][anchor]), len(b[reverse][mate])) > 100:
                continue
            possible.add(anchor)
            for candidate in b[reverse][mate]:
                witnesses[candidate].add(anchor)
    return Counter({n: len(anchors) for n, anchors in witnesses.items()}), len(possible)


def top_info(scores, target, possible):
    highest = max(scores.values(), default=0)
    winners = [n for n, value in scores.items() if value == highest]
    best = winners[0] if len(winners) == 1 and highest else None
    return dict(
        target_support=scores[target],
        mapped_neighbor_slots=possible,
        target_fraction=scores[target] / possible if possible else 0,
        best_support=highest,
        best_fraction=highest / possible if possible else 0,
        target_unique_best=best == target,
        other_unique_best=best is not None and best != target,
    )


def measure(a, b, u, v, seeds=None):
    if u not in a["nodes"] or v not in b["nodes"]:
        return dict(status="missing_or_thunk_node")
    forward, reverse, votes = seeds if seeds is not None else seed_map(a, b)
    av, ap = neighborhood_scores(a, b, u, forward)
    bv, bp = neighborhood_scores(b, a, v, reverse)
    return dict(
        status="ok",
        seed_count=len(forward),
        target_seed=forward.get(u) == v,
        target_seed_votes=votes[(u, v)],
        both_seeded_elsewhere=(u in forward and forward[u] != v and v in reverse and reverse[v] != u),
        elsewhere_seed_votes=min(votes[(u, forward[u])], votes[(reverse[v], v)])
        if u in forward and v in reverse
        else 0,
        a=top_info(av, v, ap),
        b=top_info(bv, u, bp),
    )
