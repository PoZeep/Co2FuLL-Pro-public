"""Configuration gates for the w/o CSS ablation variant."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


def all_model_route(*args, **kwargs):
    return dict(action="fewshot_then_compile", cause="ablation_no_structural_triage")


@contextmanager
def verification_gates(variant):
    if variant == "full":
        yield
        return
    if variant != "no-css":
        raise ValueError(f"Unknown variant: {variant}")
    from scripts import run_fresh_replay_v168 as replay
    from scripts import run_stable_pipeline_v203 as pipeline

    def type_gate(triage, reason):
        return triage["fewshot"] == 0 and bool(replay.type_review.TRIGGER.search(reason))

    def flow_gate(triage, reason):
        return (
            triage["fewshot"] == 0
            and not replay.flow_review.TRIGGER.search(reason)
            and bool(replay.flow_review.FLOW.search(reason))
        )

    def expanded_gate(parent, *args):
        terminal = parent["terminal"].removesuffix("_repair")
        eligible = parent["source"] == "model" and parent["prediction"] == 0
        return dict(
            fewshot=eligible and terminal == "fewshot",
            expanded=eligible and terminal in ("fewshot", "type117", "flow121"),
        )

    def tree_gate(prediction, source, *args):
        return prediction == 0 and source == "model"

    with ExitStack() as stack:
        for obj, attr, value in [
            (replay, "route", all_model_route),
            (replay.type_review, "eligible", type_gate),
            (replay.flow_review, "eligible", flow_gate),
            (pipeline, "gates181", expanded_gate),
            (pipeline, "allowed182", tree_gate),
        ]:
            stack.enter_context(patch.object(obj, attr, value))
        yield
