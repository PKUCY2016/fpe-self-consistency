"""Bounded parallel ranges from the official CIFAR URL; final official checksum is mandatory."""
import argparse
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fpe_solver.image_experiment import CIFAR_URL, verify_archive

TOTAL = 170052171
WORKERS = 8


def download(output: Path):
    if output.exists():
        return {"status": "EXISTING_VERIFIED", "md5": verify_archive(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = output.parent / "cifar-official-range-parts"
    parts.mkdir(exist_ok=True)
    started = time.monotonic()

    ranges = []
    chunk_bytes = 1024 * 1024
    # Reuse exact prefixes left by the earlier eight-range downloader.
    for region in range(8):
        region_first, region_end = region * TOTAL // 8, (region + 1) * TOTAL // 8
        legacy = parts / f"{region:02d}.part"
        for chunk, first in enumerate(range(region_first, region_end, chunk_bytes)):
            last = min(first + chunk_bytes, region_end) - 1
            path = parts / f"r{region:02d}-{chunk:02d}.part"
            expected = last - first + 1
            if not path.exists() and legacy.exists() and legacy.stat().st_size >= last - region_first + 1:
                with legacy.open("rb") as source:
                    source.seek(first - region_first)
                    data = source.read(expected)
                if len(data) == expected:
                    path.write_bytes(data)
            ranges.append((first, last, path))

    def fetch(item):
        first, last, path = item
        expected = last - first + 1
        if path.exists() and path.stat().st_size == expected:
            return path
        request = urllib.request.Request(CIFAR_URL, headers={"Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(request, timeout=60) as source:
            if source.status != 206 or source.headers.get("Content-Range") != f"bytes {first}-{last}/{TOTAL}":
                raise ValueError("official endpoint did not honor the exact range")
            size = 0
            with path.open("wb") as target:
                while block := source.read(chunk_bytes):
                    size += len(block)
                    if size > expected:
                        raise ValueError("range exceeded its declared size")
                    target.write(block)
        if size != expected:
            raise ValueError("incomplete official range")
        return path

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        paths = list(pool.map(fetch, ranges))
    merged = output.with_name(output.name + ".verified-download")
    with merged.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target)
    checksum = verify_archive(merged)
    merged.replace(output)
    return {"status": "VERIFIED", "url": CIFAR_URL, "md5": checksum,
            "bytes": TOTAL, "workers": WORKERS, "chunk_bytes": chunk_bytes, "elapsed_seconds": time.monotonic() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = download(args.output)
    args.output.with_suffix(".receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
