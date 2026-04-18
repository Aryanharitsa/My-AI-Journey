"""Download BEIR datasets to data/beir/<dataset>/.

Usage:
    python scripts/download_beir.py --datasets nfcorpus scifact fiqa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vitruvius.utils.logging import get_logger

_log = get_logger("download_beir")

DEFAULT_OUT = Path("data/beir")
DEFAULT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    try:
        from beir import util
    except ImportError:
        _log.error("beir package not installed. Run: uv pip install -e \".[dev]\"")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        url = DEFAULT_URL.format(name=name)
        _log.info("download.start dataset=%s url=%s", name, url)
        util.download_and_unzip(url, str(args.out))
        _log.info("download.done dataset=%s into=%s", name, args.out / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
