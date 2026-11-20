"""Generate HermesSim function embeddings and pairwise cosine scores."""

from co2full.runtime import asset_root, tool_path, require_server
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
import traceback

from co2full import hermes_dependencies as old

ROOT = old.ROOT
np, torch, prep = old.np, old.torch, old.prep


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = old.argparse.ArgumentParser(description=__doc__)
    for name in ("vocabulary", "input", "graphs", "binaries"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    started = time.time()
    run = Path(args.run_name).resolve()
    run.mkdir(exist_ok=False)
    (run / "runner.py").write_bytes(Path(__file__).read_bytes())
    (run / "encoder_dependency.py").write_bytes(Path(old.__file__).read_bytes())
    results, graph_hashes, binary_hashes, rebases = {}, {}, {}, {}
    try:
        vocab = json.loads(args.vocabulary.read_text())
        assert len(vocab) == 461 and set(vocab.values()) == set(range(461))
        rows = [json.loads(line) for line in args.input.open()][: args.limit]
        assert len(rows) == args.limit
        assert all(set(r) == {"pair_id", "function_a", "function_b"} for r in rows)
        functions = sorted(
            {(r[s]["binary"], r[s]["address"]) for r in rows for s in ("function_a", "function_b")}
        )
        needed = defaultdict(list)
        for key, address in functions:
            needed[key].append(address)
        directory = tool_path("hermes_checkpoint").parent.parent
        config = json.loads((directory / "config.json").read_text())
        checkpoint = tool_path("hermes_checkpoint")
        torch.set_num_threads(2)
        net = old.GGNN(
            old.EmbeddingEncoder(**config["encoder"]["embed"]),
            old.AdapMultiSoftmaxAggrV2(**config["aggr"]["msoftv2"]),
            **config["ggnn_net"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["epoch"] == 19
        net.load_state_dict(state["model_state_dict"], strict=True)
        net.eval().to("cuda:0")
        packer = object.__new__(old.GraphFactoryBase)
        write(run / "status.json", dict(state="encoding", total=len(functions), done=0))
        with (run / "embeddings.partial.jsonl").open("x") as partial, torch.inference_mode():
            for binary_index, (key, addresses) in enumerate(sorted(needed.items()), 1):
                raw = None
                try:
                    graph_path = args.graphs / (key + ".json")
                    graph_hashes[key] = sha(graph_path)
                    with graph_path.open() as source:
                        wrapper = json.load(source)
                    assert len(wrapper) == 1
                    raw = next(iter(wrapper.values()))
                    del wrapper
                    image_path = args.binaries / key / "image"
                    binary_hashes[key] = sha(image_path)
                    with image_path.open("rb") as source:
                        header = source.read(64)
                    assert header[:4] == b"\x7fELF"
                    machine = struct.unpack_from(("<" if header[5] == 1 else ">") + "H", header, 18)[0]
                    architecture = {3: "x", 62: "x", 40: "arm", 183: "arm", 8: "mips"}[machine]
                    rebase = int(raw.get("rebase_offset", 0))
                    rebases[key] = rebase
                except Exception as exc:
                    for address in addresses:
                        results[key, address] = dict(
                            binary=key, address=address, status="unavailable", error=repr(exc)
                        )
                else:
                    for address in addresses:
                        graph_address = hex(int(address, 16) + rebase)
                        try:
                            sog = raw[graph_address]["SOG"]
                            matrix, nodes = prep.create_graph(sog)
                            features = prep.create_features_matrix(nodes, sog, vocab, "SOG", architecture)
                            r, c, d, nr, nc = prep.coo2tuple(matrix)
                            mask = np.isin(((d - 1) & 3) + 1, [1, 2, 3])
                            graph = (r[mask], c[mask], d[mask], nr, nc)
                            assert len(features) == len(nodes) > 0 and int(features.max()) < 461
                            assert np.all((graph[2] >= 1) & (graph[2] <= 32))
                        except Exception as exc:
                            results[key, address] = dict(
                                binary=key, address=address, status="unavailable", error=repr(exc)
                            )
                            continue
                        batch = packer._pack_batch([graph], [features])
                        batch = tuple(v.to("cuda:0") if isinstance(v, torch.Tensor) else v for v in batch)
                        vectors = net(*batch).cpu().numpy()
                        assert vectors.shape == (1, 384) and np.isfinite(vectors).all()
                        results[key, address] = dict(
                            binary=key,
                            address=address,
                            status="ok",
                            nodes=len(features),
                            edges=len(graph[0]),
                            vector=vectors[0].tolist(),
                            graph_address=graph_address,
                        )
                        del batch, vectors, graph, features, matrix, nodes, sog, r, c, d, mask
                for address in addresses:
                    partial.write(json.dumps(results[key, address]) + "\n")
                partial.flush()
                raw = None
                gc.collect()
                write(
                    run / "status.json",
                    dict(
                        state="encoding",
                        done=len(results),
                        total=len(functions),
                        binary=key,
                        binaries_done=binary_index,
                        binaries_total=len(needed),
                        available=sum(r["status"] == "ok" for r in results.values()),
                        elapsed_seconds=time.time() - started,
                        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    ),
                )
        assert set(results) == set(functions)
        with (run / "embeddings.jsonl").open("x") as output:
            for key in functions:
                output.write(json.dumps(results[key]) + "\n")
        scores = []
        for row in rows:
            fs = [results[row[s]["binary"], row[s]["address"]] for s in ("function_a", "function_b")]
            score = None
            if all(f["status"] == "ok" for f in fs):
                a, b = (np.array(f["vector"]) for f in fs)
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                if denominator > 0:
                    score = float(np.dot(a, b) / denominator)
            scores.append(
                dict(
                    pair_id=row["pair_id"], cosine=score, status="ok" if score is not None else "unavailable"
                )
            )
        (run / "unlabeled_pair_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
        assert all(sha(args.graphs / (k + ".json")) == h for k, h in graph_hashes.items())
        assert all(sha(args.binaries / k / "image") == h for k, h in binary_hashes.items())
        write(run / "input_hashes.json", dict(graphs=graph_hashes, binaries=binary_hashes))
        manifest = dict(
            state="complete",
            pairs=len(rows),
            functions=len(functions),
            available=sum(r["status"] == "ok" for r in results.values()),
            bilateral_pairs=sum(s["status"] == "ok" for s in scores),
            elapsed_seconds=time.time() - started,
            label_fields_used=False,
            fine_tuning=False,
            output_dimension=384,
            batch_size=1,
            raw_graphs_in_memory_limit="One binary; one prepared function at a time.",
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            nonzero_rebase_binary_count=sum(v != 0 for v in rebases.values()),
            checkpoint_sha256=sha(checkpoint),
            vocabulary_sha256=sha(args.vocabulary),
            preprocessing_sha256=sha(Path(prep.__file__)),
            input_sha256=sha(args.input),
            input_hashes_sha256=sha(run / "input_hashes.json"),
            embeddings_sha256=sha(run / "embeddings.jsonl"),
        )
        write(run / "manifest.json", manifest)
        write(run / "status.json", manifest)
        print(json.dumps(manifest, indent=2), flush=True)
    except BaseException as exc:
        write(
            run / "status.json",
            dict(
                state="failed",
                error=repr(exc),
                done=len(results),
                elapsed_seconds=time.time() - started,
                traceback=traceback.format_exc(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
