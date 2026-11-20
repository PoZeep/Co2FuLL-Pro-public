"""Export pseudocode, control flow, dependencies and syntax trees for selected IDA functions."""

import importlib.util
import os
import sys

for dependency_path in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if dependency_path and dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
import json
from pathlib import Path
import traceback
import ida_auto
import ida_funcs
import ida_gdl
import ida_hexrays
import ida_loader
import ida_pro
import idaapi


def load(name):
    spec = importlib.util.spec_from_file_location(
        "co2full_export_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    request = json.loads(Path(idaapi.get_plugin_options("co2full")).read_text())
    try:
        if not ida_auto.auto_wait():
            raise RuntimeError("auto_analysis_did_not_complete")
        for plugin in request.get("decompiler_plugins", []):
            if not ida_loader.load_plugin(plugin):
                raise RuntimeError("cannot_load_decompiler_plugin:" + plugin)
        if not ida_hexrays.init_hexrays_plugin():
            raise RuntimeError(
                "hexrays_unavailable_for_this_architecture; check installation/license/plugins"
            )
        tree = load("ctree")
        dg = load("dg")
        codes, cfgs, trees, coverage = {}, [], [], []
        for address in request["addresses"]:
            ea = int(address, 16)
            func = ida_funcs.get_func(ea)
            code = dict(status="unavailable", pseudo_code="", reason="no_exact_function_entry")
            cfg = dict(
                start_ea=address,
                func_name="sub_" + address[2:],
                nodes=[],
                edges=[],
                status="unavailable",
                reason="no_exact_function_entry",
            )
            tr = dict(ok=False, function_ea=address, error="no_exact_function_entry")
            if func is not None and func.start_ea == ea:
                try:
                    blocks = list(ida_gdl.FlowChart(func))
                    starts = {b.start_ea for b in blocks}
                    cfg.update(
                        nodes=[[hex(b.start_ea), b.end_ea - b.start_ea] for b in blocks],
                        edges=[
                            [b.start_ea, s.start_ea]
                            for b in blocks
                            for s in b.succs()
                            if s.start_ea in starts
                        ],
                        status="ok" if blocks else "unavailable",
                        reason=None if blocks else "empty_cfg",
                    )
                except Exception as exc:
                    cfg["reason"] = repr(exc)
                try:
                    cfunc = ida_hexrays.decompile(ea)
                    if cfunc is None:
                        raise RuntimeError("decompile_returned_none")
                    text = tree.pseudocode_text(cfunc)
                    if not text.strip():
                        raise RuntimeError("empty_pseudocode")
                    code = dict(status="ok", pseudo_code=text, reason=None)
                    try:
                        visitor = tree.ControlTreeVisitor(cfunc)
                        visitor.apply_to(cfunc.body, None)
                        tr = dict(
                            ok=True,
                            function_ea=address,
                            pseudocode=text,
                            ctree=dict(
                                traversal="preorder_instruction_nodes",
                                nodes=visitor.nodes,
                                root_kind=tree.instruction_kind(cfunc.body),
                            ),
                        )
                    except Exception as exc:
                        tr["error"] = repr(exc)
                except Exception as exc:
                    code["reason"] = repr(exc)
                    tr["error"] = repr(exc)
            codes[address] = code
            cfgs.append(cfg)
            trees.append(tr)
            row = dict(
                address=address,
                code=code["status"],
                cfg=cfg["status"],
                ctree="ok" if tr["ok"] and tr["ctree"]["nodes"] else "unavailable",
                code_reason=code["reason"],
                cfg_reason=cfg["reason"],
                ctree_reason=tr.get("error"),
            )
            coverage.append(row)
            print("[IDA]", address, json.dumps(row), flush=True)
        outputs = request["outputs"]
        write(outputs["code"], codes)
        write(outputs["cfg"], {request["idb_key"]: cfgs})
        write(
            outputs["ctree"],
            dict(
                schema_version="hexrays_ctree_control_preorder_v1",
                input_database=request["binary"],
                selection_key=request["binary"],
                functions=trees,
            ),
        )
        dg.extract_call_graph(Path(outputs["dg"]))
        write(
            request["report"],
            dict(
                state="complete",
                ida_version=idaapi.get_kernel_version(),
                hexrays_version=ida_hexrays.get_hexrays_version(),
                functions=coverage,
                pseudocode_protocol="legacy_colored_cast_removal_v1",
            ),
        )
        return 0
    except Exception:
        write(request["report"], dict(state="failed", error=traceback.format_exc()))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    ida_pro.qexit(main())
