from pathlib import Path
import importlib.metadata as metadata
import subprocess
import sys
from . import assets
from .io import read_json, sha, write_json
from .runtime import ENGINE, RESOURCES, activate, require_server


def engine_check():
    provenance = read_json(RESOURCES / "engine-provenance.json")
    for r in provenance["files"]:
        if sha(ENGINE / r["source"]) != r["installed_sha256"]:
            raise ValueError(f"Engine changed: {r['source']}")
    return len(provenance["files"])


def check(path, model=False, full=False, out=None):
    cfg = activate(path)
    result = dict(
        engine_files=engine_check(), assets=assets.validate(cfg["assets"], full), model_checked=False
    )
    if model:
        require_server()
        expected = read_json(RESOURCES / "environment.lock.json")
        names = [
            "torch",
            "vllm",
            "transformers",
            "tokenizers",
            "numpy",
            "triton",
            "xformers",
            "vllm-flash-attn",
        ]
        actual = {n: metadata.version(n) for n in names}
        differences = {
            n: dict(expected=expected["packages"][n], actual=actual[n])
            for n in names
            if actual[n] != expected["packages"][n]
        }
        if sys.version.split()[0] != expected["python"].split()[0]:
            differences["python"] = dict(expected=expected["python"], actual=sys.version)
        if differences:
            raise ValueError(f"Inference environment differs: {differences}")
        fingerprints = read_json(RESOURCES / "model.lock.json")["files"]
        for name, r in fingerprints.items():
            p = Path(cfg["model"]) / name
            if not p.is_file() or p.stat().st_size != r["bytes"] or (full and sha(p) != r["sha256"]):
                raise ValueError(f"Model file mismatch: {name}")
        result.update(
            model_checked=True,
            model_files=len(fingerprints),
            packages=actual,
            gpu=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines(),
        )
        for tool in ["python", "probe_python"]:
            p = Path(cfg[tool])
            if not p.is_file():
                raise FileNotFoundError(f"Missing {tool}: {p}")
        probe = (
            subprocess.check_output(
                [
                    cfg["probe_python"],
                    "-c",
                    "import unicorn, elftools; print(unicorn.__version__); print(elftools.__version__)",
                ],
                text=True,
            )
            .strip()
            .splitlines()
        )
        if probe != ["2.0.1", "0.33"] and probe != ["2.0.1.post1", "0.33"]:
            raise ValueError(f"Probe versions differ: {probe}")
        result["probe_versions"] = probe
    result["state"] = "PASSED"
    if out:
        write_json(out, result)
    return result
