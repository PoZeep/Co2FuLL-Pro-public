"""Single-GPU vLLM worker serving chat completion requests."""

from co2full.runtime import (
    asset_root as _cf_asset_root,
    tool_path as _cf_tool_path,
    engine_script as _cf_engine_script,
    require_server as _cf_require_server,
)
from pathlib import Path
import argparse, asyncio, hashlib, json, os, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler


def main():
    _cf_require_server()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    assert args.port == 18171 and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1
    from vllm import LLM
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
    import importlib.metadata

    assert importlib.metadata.version("vllm") == "0.5.1" and args.dtype == "bfloat16"
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        max_model_len=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        seed=0,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    tok = llm.get_tokenizer()
    counter = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                dict(
                    pid=os.getpid(),
                    port=args.port,
                    mode="synchronous_offline_vllm",
                    prefix_caching=False,
                    max_num_seqs=32,
                ),
            )

        def do_POST(self):
            nonlocal counter
            try:
                assert self.path == "/batch"
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                jobs = body["jobs"]
                assert 0 < len(jobs) <= 32
                prompts = []
                params = []
                effective = []
                for j in jobs:
                    payload = j["payload"]
                    assert set(payload) <= {
                        "model",
                        "messages",
                        "temperature",
                        "top_p",
                        "seed",
                        "max_tokens",
                        "guided_choice",
                    }
                    assert (
                        payload["model"] == "Qwen2.5-7B-Instruct"
                        and payload["temperature"] == 0
                        and payload["top_p"] == 1
                    )
                    req = ChatCompletionRequest(**payload)
                    assert all(
                        m["role"] in ["system", "user", "assistant"]
                        and isinstance(m["content"], str)
                        and set(m) == {"role", "content"}
                        for m in payload["messages"]
                    )
                    prompt = tok.apply_chat_template(
                        conversation=payload["messages"],
                        tokenize=False,
                        add_generation_prompt=req.add_generation_prompt,
                        tools=None,
                        documents=None,
                        chat_template=req.chat_template,
                        **(req.chat_template_kwargs or {}),
                    )
                    ids = tok(prompt, add_special_tokens=req.add_special_tokens).input_ids
                    assert len(ids) == j["prompt_tokens"] and len(ids) < 32768
                    if req.max_tokens is None:
                        req.max_tokens = 32768 - len(ids)
                    assert len(ids) + req.max_tokens <= 32768
                    sp = req.to_sampling_params()
                    if req.guided_choice:
                        processor = asyncio.run(get_guided_decoding_logits_processor("outlines", req, tok))
                        assert processor is not None
                        sp.logits_processors = (sp.logits_processors or []) + [processor]
                    prompts.append(dict(prompt=prompt, prompt_token_ids=ids))
                    params.append(sp)
                    effective.append(
                        dict(
                            max_tokens=req.max_tokens,
                            seed=req.seed,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            add_special_tokens=req.add_special_tokens,
                            prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        )
                    )
                tick = time.time()
                outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
                assert len(outs) == len(jobs)
                counter += 1
                records = []
                for j, ip, eff, o in zip(jobs, prompts, effective, outs):
                    assert o.prompt_token_ids == ip["prompt_token_ids"] and len(o.outputs) == 1 and o.finished
                    z = o.outputs[0]
                    usage = dict(
                        prompt_tokens=len(o.prompt_token_ids),
                        completion_tokens=len(z.token_ids),
                        total_tokens=len(o.prompt_token_ids) + len(z.token_ids),
                    )
                    engine = dict(
                        request_id=o.request_id,
                        prompt_token_ids=o.prompt_token_ids,
                        output_token_ids=list(z.token_ids),
                        text=z.text,
                        finish_reason=z.finish_reason,
                        stop_reason=z.stop_reason,
                        finished=o.finished,
                    )
                    response = dict(
                        model="Qwen2.5-7B-Instruct",
                        choices=[
                            dict(
                                index=0,
                                message=dict(role="assistant", content=z.text),
                                finish_reason=z.finish_reason,
                            )
                        ],
                        usage=usage,
                    )
                    records.append(
                        dict(
                            pair_id=j["pair_id"],
                            method=j["method"],
                            response=response,
                            engine_output=engine,
                            effective=eff,
                        )
                    )
                self.reply(
                    200,
                    dict(
                        adapter="offline_v178",
                        batch_number=counter,
                        seconds=time.time() - tick,
                        pid=os.getpid(),
                        records=records,
                    ),
                )
            except BaseException as e:
                traceback.print_exc()
                self.reply(500, dict(error=repr(e)))

    print(json.dumps(dict(state="ready", pid=os.getpid(), port=args.port)), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
