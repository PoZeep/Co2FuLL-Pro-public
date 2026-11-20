"""Runtime configuration and path resolution."""

from pathlib import Path
import os
import sys
from .io import read_json

PACKAGE = Path(__file__).resolve().parent
ENGINE = PACKAGE / "_engine"
RESOURCES = PACKAGE / "resources"


def config(path=None):
    path = Path(path or os.environ["CO2FULL_CONFIG"]).expanduser().resolve()
    value = read_json(path)
    if value.get("schema") != "co2full-machine-v1":
        raise ValueError("Expected schema co2full-machine-v1")
    for name in [
        "assets",
        "model",
        "python",
        "java",
        "java_backend",
        "probe_python",
        "hermes_python",
        "hermes_root",
        "hermes_checkpoint",
        "hermes_vocabulary",
        "ida",
        "idb_root",
        "ida_pythonpath",
        "service_dir",
    ]:
        if value.get(name):
            p = Path(os.path.expandvars(value[name])).expanduser()
            # Resolving the Python symlink can bypass the virtual environment.
            value[name] = os.path.abspath(str(path.parent / p if not p.is_absolute() else p))
    value["config_file"] = str(path)
    return value


def activate(path):
    value = config(path)
    os.environ["CO2FULL_CONFIG"] = value["config_file"]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE.parent), str(ENGINE), str(ENGINE / "baseline"), os.environ.get("PYTHONPATH", "")]
    )
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "baseline"))
    return value


def asset_root():
    return Path(config()["assets"])


def tool_path(name):
    value = config().get(name)
    if not value:
        raise RuntimeError(f"Configure '{name}' in the machine configuration")
    return Path(value)


def engine_script(name):
    return ENGINE / "scripts" / name


def require_server():
    if sys.platform != "linux":
        raise RuntimeError(
            "Model inference and tokenization run on the Linux compute server. Local inspection is supported."
        )


def tokenizer():
    require_server()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tool_path("model")), local_files_only=True)
