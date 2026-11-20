import math

from scripts.graph_policy_v16 import evidence


def route(score, graph, frozen, direct_no=False):
    low = score[frozen["selected"]["low_source"]]
    high = score[frozen["selected"]["high_source"]]
    policy = frozen["graph_policy"]
    missing = low is None or high is None or not math.isfinite(low) or not math.isfinite(high)
    if direct_no and not missing and graph.get("status") == "ok":
        _, strong_other = evidence(graph, dict(count=2, fraction=1.0))
        any_target = (
            bool(graph.get("target_seed"))
            or graph.get("target_seed_votes", 0) > 0
            or any(graph.get(s, {}).get("target_support", 0) > 0 for s in ["a", "b"])
        )
        if low < 0 and high < 0 and strong_other and not any_target:
            return dict(action="rule_no", cause="v103_consensus_negative")
    same, different = evidence(graph, policy)
    if same and policy["action"] in ["rescue", "both"]:
        return dict(action="rule_yes", cause="graph_correspondence_support")
    if different and policy["action"] in ["veto", "both"]:
        return dict(action="fewshot", cause="graph_opposition_compile_disabled")
    if missing:
        return dict(action="fewshot_then_compile", cause="missing_density_keep_both")
    if high >= frozen["csls_rule"]["high"]:
        return dict(action="rule_yes", cause="fixed10_strong_support")
    if low >= frozen["csls_rule"]["low"]:
        return dict(action="fewshot_then_compile", cause="adaptive_allows_compile")
    return dict(action="fewshot", cause="density_weak_compile_disabled")


def decide(routing, fewshot, compile):
    action = routing["action"]
    if action == "rule_yes":
        return dict(prediction=1, source="rule_yes")
    if action == "rule_no":
        return dict(prediction=0, source="rule_no")
    if action == "fewshot" or fewshot == 1:
        return dict(prediction=fewshot, source="fewshot")
    assert action == "fewshot_then_compile" and fewshot == 0
    return dict(prediction=compile, source="compile")
