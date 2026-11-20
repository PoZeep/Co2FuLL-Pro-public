"""Command-line entry points for Co²FuLL-Pro."""

from pathlib import Path
import argparse
import json
import sys
from .io import read_json


def execute(function):
    try:
        value = function()
        if value is not None:
            print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if "--debug" in sys.argv:
            raise
        raise SystemExit(1)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--debug", action="store_true", help="Print full traceback on error")
    return p


def assets_main():
    def run():
        from .assets import validate, pack, unpack

        p = parser("Verify, package, or unpack BCSD assets.")
        sub = p.add_subparsers(dest="command", required=True)
        q = sub.add_parser("verify")
        q.add_argument("--assets", required=True)
        q.add_argument("--full", action="store_true")
        q = sub.add_parser("pack")
        q.add_argument("--assets", required=True)
        q.add_argument("--out", required=True)
        q = sub.add_parser("import")
        q.add_argument("--bundle", required=True)
        q.add_argument("--out", required=True)
        a = p.parse_args()
        if a.command == "verify":
            return validate(a.assets, a.full)
        if a.command == "pack":
            return pack(a.assets, a.out)
        return unpack(a.bundle, a.out)

    execute(run)


def doctor_main():
    def run():
        from .doctor import check

        p = parser("Check configured assets, engine, tools, and inference environment.")
        p.add_argument("--config", required=True)
        p.add_argument("--model", action="store_true")
        p.add_argument("--full", action="store_true")
        p.add_argument("--out")
        a = p.parse_args()
        return check(a.config, a.model, a.full, a.out)

    execute(run)


def data_main():
    def run():
        from .data_build import initialize, finalize, status

        p = parser("Initialize raw inputs, check extraction status, or finalize assets.")
        sub = p.add_subparsers(dest="command", required=True)
        q = sub.add_parser("init")
        q.add_argument("--out", required=True)
        q.add_argument("--from-assets")
        q.add_argument("--pairs")
        q.add_argument("--binaries")
        q.add_argument("--tools")
        q.add_argument("--examples", help="Few-shot examples JSON file")
        q = sub.add_parser("status")
        q.add_argument("--workspace", required=True)
        q = sub.add_parser("finalize")
        q.add_argument("--workspace", required=True)
        q.add_argument("--out", required=True)
        a = p.parse_args()
        if a.command == "init":
            return initialize(a.out, a.from_assets, a.pairs, a.binaries, a.tools, a.examples)
        if a.command == "status":
            return status(a.workspace)
        return finalize(a.workspace, a.out)

    execute(run)


def extract_main():
    def run():
        from .data_build import extract, EXTRACT_STEPS

        p = parser("Extract binary representations from ELF binaries.")
        p.add_argument("step", choices=EXTRACT_STEPS)
        p.add_argument("--config", required=True)
        p.add_argument("--workspace", required=True)
        p.add_argument("--binary", default="all")
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--timeout", type=int, default=1800)
        p.add_argument(
            "--plan",
            action="store_true",
            help="Write commands without executing them",
        )
        p.add_argument(
            "--retry",
            action="store_true",
            help="Retry failed or interrupted jobs",
        )
        p.add_argument(
            "--idb-manifest",
            help="Import existing IDA databases from a manifest",
        )
        return extract(p.parse_args())

    execute(run)


def features_main():
    def run():
        from .data_build import features, FEATURE_STEPS

        p = parser("Compute HermesSim embeddings, CSLS scores, graph features, and decompiler views.")
        p.add_argument("step", choices=FEATURE_STEPS)
        p.add_argument("--config", required=True)
        p.add_argument("--workspace", required=True)
        p.add_argument("--retry", action="store_true")
        p.add_argument("--timeout", type=int, default=86400)
        return features(p.parse_args())

    execute(run)


def services_main():
    def run():
        from .services import command

        p = parser("Manage local vLLM service workers.")
        p.add_argument("command", choices=["status", "start", "stop", "adopt-long"])
        p.add_argument("--config", required=True)
        p.add_argument("--pid", type=int)
        a = p.parse_args()
        return command(a.command, a.config, a.pid)

    execute(run)


def module_main(module):
    def run():
        from .pipeline import command

        p = parser(
            {
                1: "Run Calibrated Similarity Screening (CSS).",
                2: "Run initial few-shot LLM verification (Pre-Verification).",
                3: "Run evidence-guided Post-Verification.",
            }[module]
        )
        p.add_argument("--config", required=module == 1)
        p.add_argument("--run", required=True)
        p.add_argument(
            "--detach",
            action="store_true",
            help="Run worker in background",
        )
        a = p.parse_args()
        return command(module, a.run, a.config, a.detach)

    execute(run)


def route_main():
    module_main(1)


def fewshot_main():
    module_main(2)


def postverify_main():
    module_main(3)


def evaluate_main():
    def run():
        from .reporting import evaluate

        p = parser("Evaluate verification predictions against ground-truth labels.")
        p.add_argument("--run", required=True)
        p.add_argument("--labels", required=True)
        p.add_argument(
            "--retrieval-misses",
            type=int,
            default=0,
            help="Additional positives missed before verification",
        )
        p.add_argument(
            "--without-pv",
            action="store_true",
            help="Evaluate CSS and Pre-Verification only (without Post-Verification)",
        )
        a = p.parse_args()
        return evaluate(a.run, a.labels, a.retrieval_misses, a.without_pv)

    execute(run)


def inspect_main():
    def run():
        from .reporting import inspect

        p = parser("Inspect run progress or export a pair's decompiled views, prompts, and outputs.")
        p.add_argument("--run", required=True)
        p.add_argument("--pair-id")
        p.add_argument("--out")
        p.add_argument(
            "--assets",
            help="Override asset directory path",
        )
        a = p.parse_args()
        return inspect(a.run, a.pair_id, a.out, a.assets)

    execute(run)


def audit_main():
    def run():
        from .pipeline import audit

        p = parser("Verify stage outputs and reconstructed prompts against a completed run.")
        p.add_argument("--config", required=True)
        p.add_argument("--reference", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--detach", action="store_true")
        a = p.parse_args()
        return audit(a.config, a.reference, a.out, a.detach)

    execute(run)


if __name__ == "__main__":
    commands = {
        "assets": assets_main,
        "doctor": doctor_main,
        "services": services_main,
        "data": data_main,
        "extract": extract_main,
        "features": features_main,
        "route": route_main,
        "fewshot": fewshot_main,
        "postverify": postverify_main,
        "evaluate": evaluate_main,
        "inspect": inspect_main,
        "audit": audit_main,
    }
    name = sys.argv.pop(1)
    commands[name]()
