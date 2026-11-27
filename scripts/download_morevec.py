from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    # A direct script invocation puts scripts/ on sys.path instead of the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.cli import main

    raise SystemExit(main(["download-morevec", *sys.argv[1:]]))
