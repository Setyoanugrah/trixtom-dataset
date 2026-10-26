#!/usr/bin/env python3
"""Hash every kept doc in a batch's text dir (norm+SHA256, same as the
builder's dedup) and write one hex hash per line. Used on the runner
after the text phase so the NEXT batch can cross-check against this
batch without its text files, and on the VM to build v15_hashes.txt.

Usage: python hash_text_dir.py <text_dir> <out_file>
"""
import hashlib
import json
import re
import sys
from pathlib import Path


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def main():
    text_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n = 0
    seen = set()
    with open(out, "w", encoding="utf-8") as fh:
        for f in sorted(text_dir.glob("*.jsonl")):
            with open(f, encoding="utf-8") as src:
                for line in src:
                    try:
                        t = json.loads(line)["text"]
                    except Exception:
                        continue
                    hh = hashlib.sha256(norm(t).encode("utf-8")).hexdigest()
                    if hh not in seen:
                        seen.add(hh)
                        fh.write(hh + "\n")
                        n += 1
    print(f"[hash_text] {text_dir}: {n} unique hashes -> {out}")


if __name__ == "__main__":
    main()
