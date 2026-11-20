"""Single-GPU broker for standard and long-context vLLM workers."""

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import hashlib
import heapq
import itertools
import json
import os
import queue
import signal
import subprocess
import threading
import time
import traceback
import urllib.request


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Broker:
    def __init__(self, cfg, directory):
        self.cfg = cfg
        self.directory = directory
        self.events = directory / "execution-events.jsonl"
        self.tasks = queue.PriorityQueue()
        self.sequence = itertools.count()
        self.counters = {p: 0 for p in [8171, 8172, 8173]}
        self.child = None
        self.mode = None
        self.generation = 0
        self.ready = threading.Event()
        self.failure = None
        self.active = None

    def event(self, kind, **kwargs):
        value = dict(kind=kind, time=time.time(), broker_pid=os.getpid(), **kwargs)
        with self.events.open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
            f.flush()

    def gpu_empty(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(self.cfg.get("gpu", 0)),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not result.stdout.strip(), "GPU has an existing compute process; do not stop it"

    def stop_engine(self):
        if self.child is not None:
            pid = self.child.pid
            self.child.terminate()
            try:
                self.child.wait(45)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(15)
            self.event("engine_stopped", mode=self.mode, backend_pid=pid)
            self.child = None
            self.mode = None
            for _ in range(30):
                try:
                    self.gpu_empty()
                    break
                except AssertionError:
                    time.sleep(1)
            self.gpu_empty()

    def engine(self, mode):
        if self.mode == mode:
            assert self.child.poll() is None, "Physical model service exited"
            return
        self.stop_engine()
        self.gpu_empty()
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(self.cfg.get("gpu", 0)),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
        )
        if mode == "native":
            port = 18171
            args = [
                self.cfg["python"],
                "-u",
                str(Path(__file__).with_name("single_gpu_native_worker.py")),
                "--model",
                self.cfg["model"],
                "--port",
                str(port),
                "--dtype",
                "bfloat16",
            ]
        else:
            assert mode == "long"
            port = 18074
            args = [
                self.cfg["python"],
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.cfg["model"],
                "--tokenizer",
                self.cfg["model"],
                "--served-model-name",
                "Qwen2.5-7B-Instruct",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
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
        self.generation += 1
        log = self.directory / f"engine-{os.getpid()}-{self.generation:03d}-{mode}.log"
        with log.open("x") as f:
            # Keep the engine in the broker's process group for shutdown.
            self.child = subprocess.Popen(
                args, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT
            )
        self.mode = mode
        self.event(
            "engine_starting",
            mode=mode,
            backend_pid=self.child.pid,
            generation=self.generation,
            args=args,
            log=str(log),
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            assert self.child.poll() is None, f"{mode} engine exited; inspect {log}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        self.event(
                            "engine_ready", mode=mode, backend_pid=self.child.pid, generation=self.generation
                        )
                        return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"{mode} engine readiness exceeded 10 minutes")

    def submit(self, port, body):
        if self.failure:
            raise RuntimeError(self.failure)
        done = threading.Event()
        task = dict(port=port, body=body, done=done, queued=time.time())
        self.tasks.put((1 if port == 8074 else 0, next(self.sequence), task))
        if not done.wait(3500):
            raise TimeoutError("Single-GPU queued request exceeded 3500 seconds")
        if "error" in task:
            raise RuntimeError(task["error"])
        return task["response"]

    def next_task(self):
        # Give long-context requests priority before the client times out.
        with self.tasks.mutex:
            overdue = [
                (item[1], index)
                for index, item in enumerate(self.tasks.queue)
                if item[2]["port"] == 8074 and time.time() - item[2]["queued"] >= 600
            ]
            if overdue:
                _, index = min(overdue)
                item = self.tasks.queue.pop(index)
                heapq.heapify(self.tasks.queue)
                self.tasks.not_full.notify()
                return item
        return self.tasks.get()

    def work(self):
        try:
            self.engine("native")
            self.ready.set()
            while True:
                priority, order, task = self.next_task()
                # Allow native batches to arrive before switching engines.
                if priority == 1 and self.mode == "native" and time.time() - task["queued"] < 600:
                    time.sleep(0.3)
                    if not self.tasks.empty():
                        self.tasks.put((priority, order, task))
                        continue
                self.active = dict(port=task["port"], queued=task["queued"])
                port = task["port"]
                mode = "long" if port == 8074 else "native"
                self.engine(mode)
                start = time.time()
                path = "/v1/chat/completions" if mode == "long" else "/batch"
                backend_port = 18074 if mode == "long" else 18171
                request = urllib.request.Request(
                    f"http://127.0.0.1:{backend_port}{path}",
                    data=task["body"],
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=1800) as response:
                    raw = response.read()
                value = json.loads(raw)
                request_sha = sha(task["body"])
                details = dict(
                    mode=mode,
                    virtual_port=port,
                    backend_pid=self.child.pid,
                    generation=self.generation,
                    request_sha256=request_sha,
                    backend_response_sha256=sha(raw),
                    seconds=time.time() - start,
                    queued_seconds=start - task["queued"],
                )
                if mode == "native":
                    assert value["adapter"] == "offline_v178" and value["pid"] == self.child.pid
                    body = json.loads(task["body"])
                    assert 0 < len(body["jobs"]) <= 32
                    assert len(body["jobs"]) == len(value["records"])
                    self.counters[port] += 1
                    physical_batch = value["batch_number"]
                    details.update(
                        virtual_batch=self.counters[port],
                        physical_batch=physical_batch,
                        ordered_pairs=[j["pair_id"] for j in body["jobs"]],
                    )
                    for job, record in zip(body["jobs"], value["records"]):
                        assert (job["pair_id"], job["method"]) == (record["pair_id"], record["method"])
                        record["engine_output"]["single_gpu_execution"] = dict(
                            backend_pid=self.child.pid,
                            generation=self.generation,
                            physical_batch=physical_batch,
                            virtual_port=port,
                            request_sha256=request_sha,
                        )
                    # Use a stable worker identity across physical engine restarts.
                    value.update(
                        pid=os.getpid(),
                        batch_number=self.counters[port],
                        execution_mode="single_gpu_serial_native32",
                        physical_backend_pid=self.child.pid,
                    )
                else:
                    value["single_gpu_execution"] = dict(
                        backend_pid=self.child.pid, generation=self.generation, request_sha256=request_sha
                    )
                encoded = json.dumps(value, ensure_ascii=False).encode()
                self.event("request_complete", **details, forwarded_response_sha256=sha(encoded))
                task["response"] = encoded
                task["done"].set()
                self.active = None
        except BaseException as exc:
            self.failure = repr(exc)
            self.event("FAILED", error=repr(exc), traceback=traceback.format_exc())
            self.ready.set()
            if "task" in locals():
                task["error"] = self.failure
                task["done"].set()
            while not self.tasks.empty():
                _, _, pending = self.tasks.get()
                pending["error"] = self.failure
                pending["done"].set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from co2full.runtime import config

    cfg = config(args.config)
    directory = Path(cfg["service_dir"])
    broker = Broker(cfg, directory)

    def handler(port):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, value):
                data = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                ok = broker.ready.is_set() and broker.failure is None
                self.send(
                    200 if ok and self.path == "/health" else 503,
                    dict(
                        mode="single_gpu_serial_virtual_endpoint",
                        virtual_port=port,
                        pid=os.getpid(),
                        backend_mode=broker.mode,
                        backend_pid=broker.child.pid if broker.child else None,
                        active=broker.active,
                        pending=broker.tasks.qsize(),
                        error=broker.failure,
                    ),
                )

            def do_POST(self):
                try:
                    assert self.path == ("/v1/chat/completions" if port == 8074 else "/batch")
                    data = self.rfile.read(int(self.headers["Content-Length"]))
                    self.send(200, broker.submit(port, data))
                except BaseException as exc:
                    self.send(500, dict(error=repr(exc)))

        return Handler

    servers = []
    for port in [8171, 8172, 8173, 8074]:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler(port))
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=broker.work, daemon=True).start()
    while True:
        time.sleep(1)
        if broker.failure:
            broker.stop_engine()
            raise RuntimeError(broker.failure)


if __name__ == "__main__":
    main()
