"""Build the Java helper tools for decompiler view extraction."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsat", type=Path, required=True, help="GSAT 1.0 JAR")
    parser.add_argument("--font", type=Path, required=True, help="Noto Sans Regular TTF")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--javac", default="javac", help="JDK 17 javac executable")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        staging = Path(temp) / "tools"
        classes = staging / "cviews-v23"
        classes.mkdir(parents=True)
        jar = staging / "gsat-1.0-v22.jar"
        shutil.copyfile(args.gsat, jar)
        shutil.copyfile(args.font, staging / "NotoSans-Regular-v22.ttf")
        (staging / "fontconfig-v22.properties").write_text(
            "version=1\nsequence.allfonts=default\nallfonts.default=Noto Sans\n"
            "filename.Noto_Sans=NotoSans-Regular-v22.ttf\n"
        )
        java = Path(__file__).resolve().parents[1] / "tools/java/ExportCViewsV23.java"
        subprocess.run(
            [
                args.javac,
                "--release",
                "17",
                "-implicit:none",
                "-sourcepath",
                "",
                "-cp",
                str(jar),
                "-d",
                str(classes),
                str(java),
            ],
            check=True,
        )
        staging.rename(out)


if __name__ == "__main__":
    main()
