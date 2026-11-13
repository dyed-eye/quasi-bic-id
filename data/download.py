"""Download the datasets of the paper from Zenodo and check their MD5 sums.

usage:
  python data/download.py                      # every file of the record into data/
  python data/download.py disk_chain.npz       # selected files only
  python data/download.py --list               # show the files without downloading

Files that are already present with the right MD5 sum are skipped.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD_ID = "24168440"  # Zenodo record of the datasets (DOI 10.5281/zenodo.24168440)
API = "https://zenodo.org/api/records/{}"
CHUNK = 1 << 20


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def list_files(record_id: str) -> list[dict]:
    """[{name, size, md5, url}] of a published Zenodo record."""
    with urllib.request.urlopen(API.format(record_id), timeout=60) as r:
        record = json.load(r)
    files = []
    for f in record["files"]:
        name, url, checksum = f["key"], f["links"]["self"], f.get("checksum", "")
        if Path(name).name != name or ":" in name:
            raise ValueError(f"unexpected file name in the record: {name!r}")
        if not url.startswith("https://zenodo.org/") or not checksum.startswith("md5:"):
            raise ValueError(f"{name}: unexpected download link or checksum in the record")
        files.append({"name": name, "size": f["size"], "md5": checksum.removeprefix("md5:"), "url": url})
    return files


def download(file: dict, out_dir: Path) -> None:
    dst = out_dir / file["name"]
    if dst.exists() and md5sum(dst) == file["md5"]:
        print(f"{file['name']}: already present")
        return
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(file["url"], timeout=60) as r, open(part, "wb") as f:
        done = 0
        for block in iter(lambda: r.read(CHUNK), b""):
            f.write(block)
            done += len(block)
            print(f"\r{file['name']}: {done / 1e6:.0f} / {file['size'] / 1e6:.0f} MB", end="", flush=True)
    print()
    if md5sum(part) != file["md5"]:
        part.unlink()
        raise RuntimeError(f"{file['name']}: MD5 mismatch, the download is corrupted; run the script again")
    part.replace(dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Download the datasets of the paper from Zenodo.")
    p.add_argument("names", nargs="*", help="files to download (default: all)")
    p.add_argument("--record", default=RECORD_ID, help="Zenodo record id")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="target folder")
    p.add_argument("--list", action="store_true", help="list the files and exit")
    args = p.parse_args()
    if args.record is None:
        sys.exit("The datasets are not published yet (RECORD_ID is not set in data/download.py).")
    if not str(args.record).isdigit():
        sys.exit(f"a Zenodo record id is a number, got {args.record!r}")

    files = list_files(args.record)
    if args.list:
        for f in files:
            print(f"{f['name']:40s} {f['size'] / 1e6:10.1f} MB")
        return
    unknown = set(args.names) - {f["name"] for f in files}
    if unknown:
        sys.exit(f"not in the record: {', '.join(sorted(unknown))}")

    args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not args.names or f["name"] in args.names:
            download(f, args.out)


if __name__ == "__main__":
    main()
