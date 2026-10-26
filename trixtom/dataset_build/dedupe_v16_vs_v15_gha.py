#!/usr/bin/env python3
"""Cross-batch dedup check (GHA variant): drop v16 docs that duplicate v15 docs.

Same logic as dataset_build/dedupe_v16_vs_v15.py, except v15's hashes are
loaded from a precomputed hash-list file (v15_hashes.txt, one SHA256 hex
per line, built on the VM from data_v15/text) instead of scanning v15's
text files, which do not exist on the runner. v15's text phase is already
complete on the VM when this runs, so there is no wait loop.

Run AFTER the v16 text build finishes, BEFORE tokenize_pack_v16.py.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V15_HASHES = ROOT / "data_v16" / "v15_hashes.txt"
V16_TEXT = ROOT / "data_v16" / "text"
OUT = ROOT / "data_v16" / "crosscheck_v15.json"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def h(t: str) -> str:
    return hashlib.sha256(norm(t).encode("utf-8")).hexdigest()


def main():
    v15_hashes = set()
    with open(V15_HASHES, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                v15_hashes.add(line)
    print(f"[crosscheck] v15 hashes loaded: {len(v15_hashes)} from file", flush=True)

    per_file = {}
    total_kept = 0
    total_dropped = 0
    total_skipped_bad = 0
    for f in sorted(V16_TEXT.glob("*.jsonl")):
        kept_lines = []
        dropped = 0
        bad = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    t = rec["text"]
                except Exception:
                    bad += 1
                    continue
                if h(t) in v15_hashes:
                    dropped += 1
                else:
                    kept_lines.append(line)
        tmp = f.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(kept_lines)
        tmp.replace(f)
        per_file[f.name] = {"kept": len(kept_lines), "dropped_v15_dup": dropped,
                            "skipped_bad_lines": bad}
        total_kept += len(kept_lines)
        total_dropped += dropped
        total_skipped_bad += bad
        print(f"[crosscheck] {f.name}: kept={len(kept_lines)} dropped={dropped} bad={bad}", flush=True)

    result = {
        "v15_text_complete": True,
        "v15_unique_hashes": len(v15_hashes),
        "v16_kept_docs": total_kept,
        "v16_dropped_v15_duplicates": total_dropped,
        "v16_skipped_bad_lines": total_skipped_bad,
        "per_file": per_file,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("[crosscheck] FINAL " + json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
