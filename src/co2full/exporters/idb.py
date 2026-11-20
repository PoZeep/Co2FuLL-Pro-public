"""Save the database after IDA auto-analysis. Adapted from Co2FuLL-Extract's IDB exporter."""

import json
from pathlib import Path
import traceback
import ida_auto
import ida_loader
import ida_pro
import idaapi


def main():
    request = json.loads(Path(idaapi.get_plugin_options("co2full")).read_text())
    try:
        if not ida_auto.auto_wait():
            raise RuntimeError("auto_analysis_did_not_complete")
        output = Path(request["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if not ida_loader.save_database(str(output), ida_loader.DBFL_COMP):
            raise RuntimeError("save_database_failed")
        report = dict(
            state="complete",
            ida_version=idaapi.get_kernel_version(),
            input=idaapi.get_input_file_path(),
            output=str(output),
        )
        Path(request["report"]).write_text(json.dumps(report, indent=2) + "\n")
        print("[IDB] saved", output, flush=True)
        return 0
    except Exception:
        Path(request["report"]).write_text(
            json.dumps(dict(state="failed", error=traceback.format_exc())) + "\n"
        )
        return 1


if __name__ == "__main__":
    ida_pro.qexit(main())
