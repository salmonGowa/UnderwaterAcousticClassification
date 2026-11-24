"""Download the openly available data and verify SHA-256 checksums.

    python data/download.py deepship_github  --data-root <dir>   # 63 public DeepShip files (~0.74 GB)
    python data/download.py shipsear_samples --data-root <dir>   # 4 public ShipsEar sample files (84 MB zip)
    python data/download.py checksums <dataset> --data-root <dir>  # record or verify checksums of manually obtained data

The full ShipsEar and DeepShip sets are only available by email request; see data/README.md.
Checksum files live in data/checksums/<name>.sha256 and are committed. If one exists,
downloads are verified against it; otherwise it is created.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

CHECKSUM_DIR = Path(__file__).resolve().parent / "checksums"
DEEPSHIP_TREE = "https://api.github.com/repos/irfankamboh/DeepShip/git/trees/main?recursive=1"
DEEPSHIP_RAW = "https://raw.githubusercontent.com/irfankamboh/DeepShip/main/"
SHIPSEAR_SAMPLES = "https://underwaternoise.atlanttic.uvigo.es/samples.zip"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"downloaded {dest}", flush=True)


def record_or_verify(name, root, files):
    """Write data/checksums/<name>.sha256 if missing, else check every listed file against it."""
    checksum_file = CHECKSUM_DIR / f"{name}.sha256"
    actual = {str(f.relative_to(root)): sha256(f) for f in sorted(files)}
    if not checksum_file.exists():
        CHECKSUM_DIR.mkdir(exist_ok=True)
        checksum_file.write_text("".join(f"{h}  {p}\n" for p, h in actual.items()))
        print(f"recorded {len(actual)} checksums in {checksum_file}")
        return
    expected = {}
    for line in checksum_file.read_text().splitlines():
        digest, path = line.split("  ", 1)
        expected[path] = digest
    bad = [p for p, h in expected.items() if actual.get(p) != h]
    missing = sorted(set(expected) - set(actual))
    if bad or missing:
        raise SystemExit(f"checksum mismatch: {bad[:5]} missing: {missing[:5]}")
    print(f"verified {len(expected)} files against {checksum_file}")


def deepship_github(data_root):
    root = Path(data_root) / "raw" / "deepship"
    with urllib.request.urlopen(DEEPSHIP_TREE, timeout=60) as response:
        tree = json.load(response)["tree"]
    paths = [e["path"] for e in tree if e["type"] == "blob" and (e["path"].endswith(".wav") or e["path"].endswith("-metafile"))]
    for path in paths:
        download(DEEPSHIP_RAW + path, root / path)
    record_or_verify("deepship_github", root, [root / p for p in paths])


def shipsear_samples(data_root):
    root = Path(data_root) / "raw" / "shipsear"
    zip_path = Path(data_root) / "downloads" / "shipsear_samples.zip"
    download(SHIPSEAR_SAMPLES, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = Path(member).name
            if member.lower().endswith(".wav") and not name.startswith("._") and "__MACOSX" not in member:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(z.read(member))
    record_or_verify("shipsear_samples", root, list(root.glob("*.wav")))


def checksums(data_root, dataset):
    root = Path(data_root) / "raw" / dataset
    record_or_verify(f"{dataset}_full", root, [p for p in root.rglob("*") if p.is_file()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["deepship_github", "shipsear_samples", "checksums"])
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    if args.what == "checksums":
        checksums(args.data_root, args.dataset)
    else:
        globals()[args.what](args.data_root)
