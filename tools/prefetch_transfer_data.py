"""Cache byte-verified CIFAR-10 and official ResNet18 weights without using a GPU.

The Zenodo copy is documented by Eclipse Aidge's training tutorial as an exact
copy of torchvision's archive. We check the original author's published MD5
before handing any bytes to torchvision. Notebooks retain their normal official
download behavior; this optional helper only fills their local cache.

Mirror documentation: https://eclipse.dev/aidge/source/Tutorial/learning.html
Archive record: https://zenodo.org/records/15602362
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARTS = DATA / ".cifar_download_parts"
URL = "https://zenodo.org/records/15602362/files/cifar-10-python.tar.gz?download=1"
SIZE = 170_498_071
MD5 = "c58f30108f718f92721af3b95e74349a"
CHUNK = 8 * 1024 * 1024
TARGET = DATA / "cifar-10-python.tar.gz"


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_part(index):
    start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
    expected_size = end - start + 1
    destination = PARTS / f"{index:03d}.part"
    if destination.exists() and destination.stat().st_size == expected_size:
        return index, expected_size, "cached"
    for attempt in range(5):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_range = f"bytes {start}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError("Server did not return the requested archive range")
                temporary = destination.with_suffix(".downloading")
                with temporary.open("wb") as stream:
                    while block := response.read(256 * 1024):
                        stream.write(block)
                if temporary.stat().st_size != expected_size:
                    raise RuntimeError("Incomplete archive range")
                temporary.replace(destination)
            return index, expected_size, "downloaded"
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    DATA.mkdir(exist_ok=True)
    PARTS.mkdir(exist_ok=True)
    if not TARGET.exists() or checksum(TARGET) != MD5:
        count = (SIZE + CHUNK - 1) // CHUNK
        completed_bytes = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_part, index) for index in range(count)]
            for future in as_completed(futures):
                index, size, status = future.result()
                completed_bytes += size
                elapsed = time.monotonic() - started
                print(f"Part {index:02d} {status}; {completed_bytes / SIZE:.1%}; "
                      f"{completed_bytes / 1024**2:.1f} MB in {elapsed:.0f}s", flush=True)
        assembled = DATA / "cifar-10-python.verified.tmp"
        with assembled.open("wb") as output:
            for index in range(count):
                with (PARTS / f"{index:03d}.part").open("rb") as part:
                    for block in iter(lambda: part.read(1024 * 1024), b""):
                        output.write(block)
        if assembled.stat().st_size != SIZE or checksum(assembled) != MD5:
            raise RuntimeError("Archive failed the ORIGINAL official MD5; it will not be used")
        assembled.replace(TARGET)
        print("Original official archive MD5 verified:", MD5, flush=True)
        # Remove only this helper's known, successfully verified part files.
        for index in range(count):
            (PARTS / f"{index:03d}.part").unlink()
        PARTS.rmdir()
    from torchvision.datasets import CIFAR10
    from torchvision.models import ResNet18_Weights
    CIFAR10(DATA, train=True, download=True)
    CIFAR10(DATA, train=False, download=True)
    print("CIFAR-10 extracted and individual files verified", flush=True)
    ResNet18_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    print("Official ResNet-18 weights downloaded and hash verified", flush=True)


if __name__ == "__main__":
    main()
