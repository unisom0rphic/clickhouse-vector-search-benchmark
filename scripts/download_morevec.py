from __future__ import annotations

from pathlib import Path

import gdown

URL = "https://drive.google.com/drive/folders/1AqAVI8ASROqrFCQdEMPB8RNzPwilijRp?usp=drive_link"


def main() -> int:
    output = Path("data").resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download_folder(url=URL, output=str(output), quiet=False)
    if not downloaded:
        raise SystemExit("MoReVec download failed")
    print(f"Downloaded MoReVec under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
