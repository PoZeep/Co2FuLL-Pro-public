import idaapi
import idautils
import idc
import ida_pro
import time
import networkx as nx
import pickle
from pathlib import Path


def get_func_tags(func_ea):
    flags = idc.get_func_attr(func_ea, idc.FUNCATTR_FLAGS)
    tags = []
    if flags & idc.FUNC_NORET:
        tags.append("FUNC_NORET")
    if flags & idc.FUNC_FAR:
        tags.append("FUNC_FAR")
    if flags & idc.FUNC_USERFAR:
        tags.append("FUNC_USERFAR")
    if flags & idc.FUNC_LIB:
        tags.append("FUNC_LIB")
    if flags & idc.FUNC_STATIC:
        tags.append("FUNC_STATIC")
    if flags & idc.FUNC_FRAME:
        tags.append("FUNC_FRAME")
    if flags & idc.FUNC_HIDDEN:
        tags.append("FUNC_HIDDEN")
    if flags & idc.FUNC_THUNK:
        tags.append("FUNC_THUNK")
    if flags & idc.FUNC_BOTTOMBP:
        tags.append("FUNC_BOTTOMBP")
    return tags


class IEViewer(object):
    def __init__(self):
        self._imports = []
        self._exports = []

    def imports_names_cb(self, ea, name, ord):
        tmp = name.split("@@")
        if len(tmp) == 1:
            self._imports.append([ord, ea, tmp[0], ""])
        else:
            self._imports.append([ord, ea, tmp[0], tmp[1]])
        return True

    def get_imports(self, only_name=False):
        if self._imports:
            return [item[2:] for item in self._imports] if only_name else self._imports

        nimps = idaapi.get_import_module_qty()
        for i in range(nimps):
            idaapi.enum_import_names(i, self.imports_names_cb)
        self._imports.sort(key=lambda x: x[2])
        return [item[2:] for item in self._imports] if only_name else self._imports

    def get_exports(self, only_name=False):
        if self._exports:
            return [item[3] for item in self._exports] if only_name else self._exports
        self._exports = list(idautils.Entries())
        return [item[3] for item in self._exports] if only_name else self._exports


def _add_node_edge(call_graph, callee_ea, called_ea):
    caller = idaapi.get_func(called_ea)
    if caller is not None and caller.start_ea != callee_ea:
        if hex(caller.start_ea) not in call_graph.nodes:
            call_graph.add_node(
                hex(caller.start_ea),
                value=idaapi.get_func_name(caller.start_ea),
                type="func",
                tags=get_func_tags(caller.start_ea),
            )
        edge = hex(caller.start_ea), hex(callee_ea)
        if edge not in call_graph.edges:
            call_graph.add_edge(edge[0], edge[1], called_ea=[hex(called_ea)])
        else:
            call_graph[edge[0]][edge[1]]["called_ea"].append(hex(called_ea))


def restore_functions():
    """Find unrecognized branch targets in the text segment."""
    text_seg = idaapi.get_segm_by_name(".text")
    print(f"Scanning .text segment: {hex(text_seg.start_ea)} - {hex(text_seg.end_ea)}")

    start_func_eas = []
    ea = text_seg.start_ea
    while ea < text_seg.end_ea:
        if not idaapi.get_func(ea):
            for ref_ea in list(idautils.CodeRefsTo(ea, 0)):
                print_insn_mnem = getattr(idc, "print_insn_mnem", None) or getattr(idaapi, "print_insn_mnem")
                if print_insn_mnem(ref_ea) in {"B", "BL"}:
                    print(f"Found ea {hex(ea)} with code ref")
                    start_func_eas.append(ea)

        ea = idc.next_head(ea)
    print(f"Total {len(start_func_eas)} being identified")
    return start_func_eas


def extract_call_graph(save_path):
    start = time.time()
    call_graph = nx.DiGraph()

    ie_viewer = IEViewer()
    for func_ea in idautils.Functions():
        func_name = idaapi.get_func_name(func_ea)
        if func_name is None:
            continue
        func_tags = get_func_tags(func_ea)
        call_graph.add_node(hex(func_ea), value=func_name, type="func", tags=func_tags)
        for called_ea in idautils.CodeRefsTo(func_ea, 1):
            # called_ea is the call site in the caller.
            _add_node_edge(call_graph, callee_ea=func_ea, called_ea=called_ea)

    for str_item in idautils.Strings():
        call_graph.add_node(hex(str_item.ea), value=str(str_item), type="str", tags=[])
        for called_ea in idautils.DataRefsTo(str_item.ea):
            _add_node_edge(call_graph, callee_ea=str_item.ea, called_ea=called_ea)

    call_graph.imports = ie_viewer.get_imports(only_name=False)
    call_graph.exports = ie_viewer.get_exports(only_name=False)
    call_graph.time_cost = time.time() - start
    call_graph.name = idc.get_root_filename()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(call_graph, f)
    print(f"[+] Dependency graph saved in {save_path}")


if __name__ == "__main__":
    if not idaapi.get_plugin_options("DG"):
        print("[!] -ODG option is missing")
        ida_pro.qexit(1)

    plugin_options = idaapi.get_plugin_options("DG").split(":")
    print(plugin_options)
    save_path = Path(plugin_options[0])
    extract_call_graph(save_path)
    ida_pro.qexit(0)
