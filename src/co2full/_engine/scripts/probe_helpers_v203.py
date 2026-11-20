from co2full.runtime import require_server as _cf_require_server

from pathlib import Path

import json, sys

from concurrent.futures import ProcessPoolExecutor

from scripts.prepare_helper_observations_v195 import fresh


def main():
    _cf_require_server()
    jobs = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    with dest.open("x") as h, ProcessPoolExecutor(max_workers=4) as pool:
        for r in pool.map(fresh, jobs):
            o = r["observation"]
            vs = {int(v["address"], 16): v for v in r["validation"]["helpers"]}
            for x in o["helpers"]:
                h.write(
                    json.dumps(
                        dict(
                            binary=o["binary"],
                            address=x["address"],
                            image_sha256=o["image_sha256"],
                            origin="v203_fresh_finite_probes",
                            observation=x,
                            validation=vs.get(int(x["address"], 16)),
                        )
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
