"""Export Hex-Rays control trees for selected function addresses."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time

import ida_hexrays
import ida_pro
import idaapi


CAST_PATTERN = re.compile(r"\(\x02\t\x01\x17.+?\x02\x17\x01\t\)")


def normalize_ea(value):
    return hex(int(str(value), 16)).lower()


def ea_or_none(item):
    ea = getattr(item, "ea", idaapi.BADADDR)
    return None if ea == idaapi.BADADDR else hex(ea)


def item_text(item, cfunc):
    printer = getattr(item, "print1", None)
    if printer is not None:
        for args in ((cfunc,), ()):
            try:
                rendered = printer(*args)
                if rendered is not None:
                    text = str(rendered).strip()
                    if text:
                        return idaapi.tag_remove(text)
            except Exception:
                pass
    debug_string = getattr(item, "dstr", None)
    if debug_string is not None:
        try:
            text = str(debug_string()).strip()
            if text:
                return idaapi.tag_remove(text)
        except Exception:
            pass
    return ""


def instruction_kind(insn):
    names = {
        ida_hexrays.cit_empty: "empty",
        ida_hexrays.cit_expr: "expression",
        ida_hexrays.cit_block: "block",
        ida_hexrays.cit_if: "if",
        ida_hexrays.cit_for: "for",
        ida_hexrays.cit_while: "while",
        ida_hexrays.cit_do: "do_while",
        ida_hexrays.cit_switch: "switch",
        ida_hexrays.cit_break: "break",
        ida_hexrays.cit_continue: "continue",
        ida_hexrays.cit_return: "return",
        ida_hexrays.cit_goto: "goto",
        ida_hexrays.cit_asm: "asm",
    }
    return names.get(insn.op, f"instruction_{insn.op}")


def condition_text(insn, cfunc):
    try:
        if insn.op == ida_hexrays.cit_if:
            return item_text(insn.cif.expr, cfunc)
        if insn.op == ida_hexrays.cit_for:
            return item_text(insn.cfor.expr, cfunc)
        if insn.op == ida_hexrays.cit_while:
            return item_text(insn.cwhile.expr, cfunc)
        if insn.op == ida_hexrays.cit_do:
            return item_text(insn.cdo.expr, cfunc)
        if insn.op == ida_hexrays.cit_switch:
            return item_text(insn.cswitch.expr, cfunc)
    except Exception:
        return ""
    return ""


def loop_header(insn, cfunc):
    if insn.op != ida_hexrays.cit_for:
        return {}
    fields = {}
    try:
        init = item_text(insn.cfor.init, cfunc)
        step = item_text(insn.cfor.step, cfunc)
        if init:
            fields["init"] = init
        if step:
            fields["step"] = step
    except Exception:
        pass
    return fields


class ControlTreeVisitor(ida_hexrays.ctree_parentee_t):
    """Record instruction nodes in preorder."""

    def __init__(self, cfunc):
        super().__init__(ida_hexrays.CV_FAST)
        self.cfunc = cfunc
        self.nodes = []
        self._last_at_depth = {}

    def visit_insn(self, insn):
        depth = len(self.parents)
        for known_depth in list(self._last_at_depth):
            if known_depth >= depth:
                del self._last_at_depth[known_depth]
        parent_id = self._last_at_depth.get(depth - 1)
        node_id = len(self.nodes)
        node = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "kind": instruction_kind(insn),
            "ea": ea_or_none(insn),
        }
        if insn.op == ida_hexrays.cit_expr or insn.op in {
            ida_hexrays.cit_return,
            ida_hexrays.cit_break,
            ida_hexrays.cit_continue,
            ida_hexrays.cit_goto,
            ida_hexrays.cit_asm,
        }:
            node["text"] = item_text(insn, self.cfunc)
        condition = condition_text(insn, self.cfunc)
        if condition:
            node["condition"] = condition
        node.update(loop_header(insn, self.cfunc))
        self.nodes.append(node)
        self._last_at_depth[depth] = node_id
        return 0


def pseudocode_text(cfunc):
    lines = []
    for line in cfunc.get_pseudocode():
        lines.append(idaapi.tag_remove(CAST_PATTERN.sub("", line.line)))
    return "\n".join(lines)


def export_function(func_ea):
    started = time.time()
    cfunc = ida_hexrays.decompile(func_ea)
    if cfunc is None:
        raise RuntimeError("decompile_returned_none")
    visitor = ControlTreeVisitor(cfunc)
    visitor.apply_to(cfunc.body, None)
    return {
        "function_ea": hex(func_ea),
        "function_name": idaapi.get_func_name(func_ea),
        "pseudocode": pseudocode_text(cfunc),
        "ctree": {
            "traversal": "preorder_instruction_nodes",
            "root_kind": instruction_kind(cfunc.body),
            "nodes": visitor.nodes,
        },
        "decompile_seconds": round(time.time() - started, 6),
    }


def main():
    options = idaapi.get_plugin_options("ctree")
    if not options:
        print("[!] -Octree:OUTPUT_JSON:SELECTION_JSON:SELECTION_KEY is required")
        return 1
    parts = options.split(":", 2)
    if len(parts) != 3:
        print("[!] expected -Octree:OUTPUT_JSON:SELECTION_JSON:SELECTION_KEY")
        return 1
    output_path, selection_path, selection_key = parts
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selected = selection.get(selection_key)
    if not isinstance(selected, list) or not selected:
        print(f"[!] selection key {selection_key!r} has no function addresses")
        return 1

    results = []
    for raw_ea in selected:
        ea = int(normalize_ea(raw_ea), 16)
        try:
            results.append({"ok": True, **export_function(ea)})
            print(f"[+] exported ctree {hex(ea)}")
        except ida_hexrays.DecompilationFailure as exc:
            results.append({"ok": False, "function_ea": hex(ea), "error": f"decompilation_failure:{exc}"})
        except Exception as exc:
            results.append({"ok": False, "function_ea": hex(ea), "error": f"{type(exc).__name__}:{exc}"})

    output = {
        "schema_version": "hexrays_ctree_control_preorder_v1",
        "input_database": idaapi.get_input_file_path(),
        "selection_key": selection_key,
        "functions": results,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[+] ctree saved in {path}")
    return 0


if __name__ == "__main__":
    ida_pro.qexit(main())
