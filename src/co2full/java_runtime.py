"""Java launcher wrapper with headless font configuration."""

from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

from .runtime import config


def relocate(text, font_directory):
    def replace(match):
        candidate = Path(font_directory) / Path(match[2].strip()).name
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing bundled headless font: {candidate}")
        return match[1] + str(candidate.resolve())

    return re.sub(r"(?m)^(filename\.[^=]+=)(.+)$", replace, text)


def main():
    cfg = config()
    value = cfg.get("java_backend")
    if not value:
        raise RuntimeError("Configure java_backend as the real Java17 executable")
    backend = Path(os.path.expandvars(value)).expanduser()
    if not backend.is_absolute():
        raise ValueError("java_backend must be absolute so it remains valid in run configuration snapshots")
    backend = backend.absolute()
    if not backend.is_file() or backend.name == "co2full-java":
        raise ValueError("java_backend must point to real Java, not this wrapper")
    args = sys.argv[1:]
    fonts = next((x.split("=", 1)[1] for x in args if x.startswith("-Dsun.java2d.fontpath=")), None)
    font_config = next((x.split("=", 1)[1] for x in args if x.startswith("-Dsun.awt.fontconfig=")), None)
    with tempfile.TemporaryDirectory(prefix="co2full-java-fonts-") as tmp:
        if font_config:
            if not fonts:
                raise ValueError("Font configuration requires an explicit bundled font directory")
            generated = Path(tmp) / "fontconfig.properties"
            generated.write_text(relocate(Path(font_config).read_text(), fonts))
            args = [
                "-Dsun.awt.fontconfig=" + str(generated) if x.startswith("-Dsun.awt.fontconfig=") else x
                for x in args
            ]
        raise SystemExit(subprocess.call([str(backend), *args]))


if __name__ == "__main__":
    main()
