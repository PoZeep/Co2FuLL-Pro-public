"""Process management for local vLLM inference workers."""

from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from .io import read_json, write_json, lock, sha
from .runtime import activate, ENGINE, require_server


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else dict(pid=pid, start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_args(pid):
    return [x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def current(cfg):
    path = Path(cfg["service_dir"]) / "workers.json"
    return read_json(path) if path.exists() else {}


def verify(cfg):
    records = current(cfg)
    if set(records) != {"8171", "8172", "8173", "8074"}:
        raise RuntimeError("Start three native workers and the long-context service first")
    for port, record in records.items():
        if identity(record["identity"]["pid"]) != record["identity"] or not healthy(int(port)):
            raise RuntimeError(f"Worker {port} is missing or changed")
        if process_args(record["identity"]["pid"]) != record["args"]:
            raise RuntimeError(f"Worker {port} command changed")
    return {p: r["identity"] for p, r in records.items()}


def stop_record(record):
    who = record["identity"]
    if identity(who["pid"]) != who:
        return
    # Adopted workers can share a process group with unrelated processes.
    group = record.get("process_group")
    if group and os.getpgid(who["pid"]) == group:
        os.killpg(group, signal.SIGTERM)
    else:
        os.kill(who["pid"], signal.SIGTERM)
    for _ in range(45):
        if identity(who["pid"]) != who:
            return
        time.sleep(1)
    if identity(who["pid"]) == who:
        if group and os.getpgid(who["pid"]) == group:
            os.killpg(group, signal.SIGKILL)
        else:
            os.kill(who["pid"], signal.SIGKILL)


def start_single_gpu(cfg):
    from . import services
    from co2full.io import write_json

    directory = Path(cfg["service_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for port in [8171, 8172, 8173, 8074, 18171, 18074]:
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0, f"Existing listener on {port}"
    args = [
        cfg["python"],
        "-u",
        str(Path(__file__).with_name("single_gpu_broker.py")),
        "--config",
        cfg["config_file"],
    ]
    log = directory / "broker.log"
    with log.open("a") as f:
        process = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
    record = dict(
        identity=services.identity(process.pid),
        args=args,
        log=str(log),
        process_group=process.pid,
        mode="single_gpu_serial_virtual_endpoint",
    )
    write_json(
        directory / "workers.json",
        {str(port): dict(record, virtual_port=port) for port in [8171, 8172, 8173, 8074]},
    )
    deadline = time.monotonic() + 660
    while time.monotonic() < deadline:
        assert process.poll() is None, f"Single-GPU broker exited; inspect {log}"
        if all(services.healthy(port) for port in [8171, 8172, 8173, 8074]):
            return services.verify(cfg)
        time.sleep(2)
    raise TimeoutError("Single-GPU broker readiness exceeded 11 minutes")


def command(action, path, pid=None):
    require_server()
    cfg = activate(path)
    out = Path(cfg["service_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (
        cfg.get("native_ports", [8171, 8172, 8173]) != [8171, 8172, 8173]
        or cfg.get("long_port", 8074) != 8074
    ):
        raise ValueError("The validated protocol uses native ports 8171-8173 and long port 8074")
    with lock(out / "management.lock"):
        records = current(cfg)
        if action == "status":
            return {
                p: dict(
                    identity=r["identity"],
                    alive=identity(r["identity"]["pid"]) == r["identity"],
                    healthy=healthy(int(p)),
                )
                for p, r in records.items()
            }
        if action == "adopt-long":
            if pid is None:
                raise ValueError("--pid is required")
            args = process_args(pid)
            required = {
                "--model": cfg["model"],
                "--port": "8074",
                "--dtype": "bfloat16",
                "--max-model-len": "131072",
                "--max-num-seqs": "2",
            }
            if (
                "vllm.entrypoints.openai.api_server" not in args
                or "--enable-prefix-caching" in args
                or "--enforce-eager" not in args
            ):
                raise ValueError("This process does not match the validated long-context service")
            for k, v in required.items():
                if k not in args or args[args.index(k) + 1] != v:
                    raise ValueError(f"Long service differs: {k}")
            if Path(f"/proc/{pid}").stat().st_uid != os.getuid() or not healthy(8074):
                raise ValueError("Long service must be owned by the current user and healthy")
            records["8074"] = dict(identity=identity(pid), args=args, adopted=True)
            write_json(out / "workers.json", records)
            return dict(state="ADOPTED", port=8074, identity=identity(pid))
        if action == "stop":
            for r in {r["identity"]["pid"]: r for r in records.values()}.values():
                stop_record(r)
            write_json(out / "workers.stopped.json", records)
            write_json(out / "workers.json", {})
            return dict(state="STOPPED")
        if cfg.get("execution", "single-gpu") == "single-gpu":
            if records and any(identity(r["identity"]["pid"]) == r["identity"] for r in records.values()):
                return verify(cfg)
            return start_single_gpu(cfg)
        if cfg["execution"] != "multi-gpu":
            raise ValueError("execution must be single-gpu or multi-gpu")
        for port in [8171, 8172, 8173, 8074]:
            old = records.get(str(port))
            if old and identity(old["identity"]["pid"]) == old["identity"]:
                if not healthy(port):
                    raise RuntimeError(f"Recorded service {port} is unhealthy; inspect its log")
                continue
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(
                        f"Port {port} is occupied by an unregistered process; it was not stopped"
                    )
            env = dict(
                os.environ,
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="4",
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
            )
            if port != 8074:
                env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("native_gpus", [0, 1, 2])[port - 8171])
                args = [
                    cfg["python"],
                    "-u",
                    str(ENGINE / "scripts/offline_worker_v178.py"),
                    "--model",
                    cfg["model"],
                    "--port",
                    str(port),
                    "--dtype",
                    "bfloat16",
                ]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("long_gpu", 3))
                args = [
                    cfg["python"],
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    cfg["model"],
                    "--tokenizer",
                    cfg["model"],
                    "--served-model-name",
                    "Qwen2.5-7B-Instruct",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8074",
                    "--dtype",
                    "bfloat16",
                    "--max-model-len",
                    "131072",
                    "--tensor-parallel-size",
                    "1",
                    "--gpu-memory-utilization",
                    "0.85",
                    "--max-num-seqs",
                    "2",
                    "--seed",
                    "0",
                    "--disable-log-requests",
                    "--rope-scaling",
                    json.dumps(dict(type="yarn", factor=4.0, original_max_position_embeddings=32768)),
                    "--enable-chunked-prefill",
                    "--max-num-batched-tokens",
                    "8192",
                    "--enforce-eager",
                ]
            logfile = out / f"{port}-{int(time.time())}.log"
            with logfile.open("x") as h:
                proc = subprocess.Popen(
                    args,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=h,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            records[str(port)] = dict(
                identity=identity(proc.pid), args=args, log=str(logfile), process_group=proc.pid
            )
            write_json(out / "workers.json", records)
            print(f"[SERVICE] starting port={port}, pid={proc.pid}, log={logfile}", flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            for port, r in records.items():
                if identity(r["identity"]["pid"]) != r["identity"]:
                    raise RuntimeError(f"Worker {port} exited; inspect {r.get('log')}")
            if all(healthy(int(p)) for p in records):
                return dict(state="READY", identities=verify(cfg))
            time.sleep(3)
        raise TimeoutError("Services did not become ready; logs preserved")
