#!/usr/bin/env python3
"""trixtom-1.0-magorch — full dataset v16 text builder (streaming, resumable).

v16 is the eleventh 1 BILLION-token batch (session 2, even batch).
Third batch with the NEW SOURCES (HPLT-id + Taiga-ru) researched
2026-10-26 (research_new_sources.md). Same targets as v12/v14:
~1.0B tokens (id ~60% / ru ~40%), i.e. ~4.35B kept chars.

WIKIPEDIA-RU NOTE (v14): EXHAUSTED — all 21 dump files (00000-00020)
of wikimedia/wikipedia 20231101.ru have been read at least once
(v10 read the last unread tails, files 00015-00018). No newer dump
exists on HF. Wikipedia is NOT used in v14 and is REMOVED from the
source list entirely. Wikipedia-id was 100% consumed by v2 and is
not used either.

NEW SOURCES (v14): HPLT 2.0 cleaned ind_Latn (Indonesian, streaming
via datasets, 98M rows — effectively inexhaustible) + Taiga proza
(Russian fiction, parquet_remote, 83 files, 0% measured overlap vs
v1-v10) + mC4 capped slices as before. FineWeb-2 is FALLBACK /
OVERFLOW ONLY in v14 (kept so a shortfall still fills); it should
contribute ~0 if HPLT+Taiga hit their caps.

MANDATORY SPAM FILTER (user requirement, 2026-10-26): HPLT samples
contained gambling SEO spam ("Situs Judi Slot Online...") that
clean_doc() keeps. A denylist filter (SPAM_DENYLIST) is applied to
every raw doc BEFORE clean_doc(); matches are dropped and counted
in progress.json as spam_dropped. The filter was unit-tested
(test_spam_v12.py) BEFORE the main HPLT run: the known spam sample
is dropped and 20 clean Indonesian docs are kept.

FIXES vs build_text_v6.py — this file exists because v6 stalled twice on
FineWeb-2 (see DIAGNOSIS below); do NOT regress them:
  1. FineWeb-2 NO LONGER uses datasets streaming + blind skip-ahead.
     v6's skip loop re-iterated STREAM_SKIP + already-processed docs from
     the start of the stream on EVERY --resume (~1,500 docs/s, so a
     2.35M-doc skip cost ~26 min of total silence: no log line, no
     progress.json write, no counter increment). The skip is slow
     because it is BANDWIDTH-bound: HF serves this VM at ~2.5 MB/s
     (measured 2026-10-25: curl 100 MB in 41 s; hf_hub_download
     ~2 MB/s), and every resume re-downloaded gigabytes of parquet
     data just to throw it away. v7 instead reads each FineWeb-2
     parquet file REMOTELY with pyarrow over fsspec HTTP range reads,
     seeking by ROW GROUP — bytes are fetched once, processing
     overlaps the transfer, and resume = reopen the current file at
     the saved row (seconds). The v1-v10 consumed prefix is walked by
     reading parquet FOOTERS only (HTTP range reads, no download) and
     skipping whole files whose rows are fully consumed. If a remote
     open fails, it falls back to a resumable hf_hub_download of that
     one file (~4.85 GB, one file on disk at a time).
  2. Heartbeat: a watchdog thread prints a line every 120 s with the
     current phase and seconds since last activity, and every phase
     (hash rebuild, skip, process, parquet walk) saves progress
     regularly. A silent builder is now visible within 2 minutes.
  3. Stream watchdog: streaming reads (mC4) are wrapped in a SIGALRM
     timeout (600 s). A hung HTTP stream raises instead of blocking
     forever; progress is saved and the process exits non-zero.
  4. BUG FIX: v6 marked a source done=True even when its try/except
     caught an error (partial source silently treated as finished).
     v7 only marks done on genuine finish (cap / target / exhausted);
     on error it saves progress, leaves done=False, and exits non-zero
     so the supervisor retries it.
  5. The builder now logs to data_v16/build_text_v16.log in APPEND mode
     with UTC timestamps (v6's log was overwritten with `>` on every
     resume, destroying the evidence of earlier runs).
  6. rebuild_hashes() now prints progress per directory (v6 did it
     silently; it takes minutes over ~20 GB of text on every resume).

DIAGNOSIS (v6 stalls, reconstructed 2026-10-25 from build_text_v6.py,
data_v6/progress.json and data_v6/build_text_v6.log — the v6 agent left
no final report and its per-run logs were overwritten):
  * Stall 1 (~20 min, 18:41-19:19 WIB): fineweb2-id progress frozen at
    2,782,720,293 chars / 569,471 kept docs. Cause: a --resume re-ran
    rebuild_hashes() (silent, minutes) + the blind skip of
    1,649,861 + ~400k processed docs at ~1,458 docs/s (measured
    2026-10-25: 20,000 docs in 13.7 s) = ~24 min with zero output.
  * Stall 2 (~1h, 19:56-20:41 WIB): same, but the skip had grown to
    ~2.27M docs (~26 min) and the agent resumed the builder TWICE
    (PIDs seen at 12:45 and 13:22 UTC), so two silent skip phases
    stacked. The builder was never actually hung — it was working,
    invisibly. The "stall" was a visibility + resume-cost defect, not
    a network failure: no HTTP error or exception appears anywhere in
    the surviving log, and the final run's fineweb2-ru skip of
    498,525 docs took ~299 s and then processed normally.
  * Contributing: unauthenticated HF Hub requests (rate limits) and
    streaming re-downloading shards on every resume (no full parquet
    file is ever cached by datasets streaming).

Sources (v16: HPLT-id + Taiga-ru + mC4; FineWeb-2 fallback only):
  1. allenai/c4 id / ru (mC4, capped slice, streaming — skip is small,
     ~107k/253k docs after v12, kept with heartbeat + watchdog).
  2. HPLT/HPLT2.0_cleaned ind_Latn via datasets streaming
     (row-counter resume, start row 1,093,525, cap = id target 2,600M chars).
     SPAM FILTER APPLIES (see above). First-row latency can be
     minutes (140-475 files per shard dir) — do not mistake for stall.
  3. cointegrated/taiga_stripped_proza via parquet_remote (same
     pyarrow + fsspec range-reader as FineWeb-2, per-row-group
     checkpoints). 83 parquet files, column "text", resume file 00012
     row 8,852 (v14 end state), cap = ru target 1,750M chars.
  4. HuggingFaceFW/fineweb-2 ind_Latn / rus_Cyrl via parquet_remote
     (see fix 1) — FALLBACK ONLY, skip continues from v14's end
     (6,439,171 / 2,117,617; v14 consumed 0 fw2 rows; v15's fw2
     consumption is added at launch by watch_v16_launch.sh, MAX rule).

NOT attempted: oscar-corpus/OSCAR-2301 (gated, no token) and
uonlp/CulturaX (gated too) — no token/secret exists in any file.

Cleaning / dedup: IDENTICAL to v1-v10 (whitespace collapsed, docs
200..12000 chars, longer split at paragraph boundaries into
<=6000-char chunks, >=60% letters, <=20% digits, <=30% duplicate
lines, script check). Dedup is against v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13 AND v14
(disk hash cache, extended in v16 with v13+v14 text — see
_extend_with_extra_dirs); v15 overlap is checked before packing
(separate cross-check script), because v15 builds in parallel with v16.

Output: data_v16/text/{lang}-{shard:05d}.jsonl; progress.json updated
regularly. Usage: python build_text_v16.py [--resume] [--only mc4-id,...]
"""
import argparse, hashlib, json, os, re, signal, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_v16" / "text"
PROGRESS = ROOT / "data_v16" / "progress.json"
LOGFILE = ROOT / "data_v16" / "build_text_v16.log"
SHARD_DOCS = 50000
SHARD_BYTES = 250_000_000

CACHE = ROOT / "data_v16" / "cache"  # unused in v16 (Wikipedia exhausted)
PARQUET_CACHE = ROOT / "data_v16" / "cache_parquet"  # one remote parquet file at a time (fallback download)

# (lang, source_name, kind, repo, config, cap_chars_or_None, parquet_prefix_or_None)
# Order matters: mC4 caps first, then the main new sources fill the lang
# targets; FineWeb-2 fallback runs last and is skipped once targets are met.
SOURCES = [
    ("id", "mc4", "stream", "allenai/c4", "id", 50_000_000, None),
    ("ru", "mc4", "stream", "allenai/c4", "ru", 100_000_000, None),
    ("id", "hplt", "stream", "HPLT/HPLT2.0_cleaned", "ind_Latn", None, None),
    ("ru", "taiga", "parquet_remote", "cointegrated/taiga_stripped_proza", "", None, "data/"),
    ("id", "fineweb2", "parquet_remote", "HuggingFaceFW/fineweb-2", "ind_Latn", None, "data/ind_Latn/train/"),
    ("ru", "fineweb2", "parquet_remote", "HuggingFaceFW/fineweb-2", "rus_Cyrl", None, "data/rus_Cyrl/train/"),
]

# Raw docs already consumed from the streams by v1..v15 (see below).
# v16 resume offsets (derived from v14 FINAL-REPORT §12 + data_v14/progress.json):
# fineweb2 id: v14 skip 6,439,171 + v14 processed 0 = 6,439,171
# fineweb2 ru: v14 skip 2,117,617 + v14 processed 0 = 2,117,617
# mc4 id: v14 skip 106,680 + v14 processed 17,678 = 124,358
# mc4 ru: v14 skip 252,620 + v14 processed 33,961 = 286,581
# hplt-id: v14 skip 546,566 + v14 processed 546,959 = 1,093,525 total consumed
# NOTE: watch_v16_launch.sh re-finalizes the fw2/mc4 skips at launch
# with the MAX rule against v15's progress.json (v15 consumes fw2+mc4
# in parallel); it patches STREAM_SKIP in this file before launching.
# GHA: skips finalised on the VM (v15 consumption added to v15 launch bases).
STREAM_SKIP = {"fineweb2-id": 7_137_717, "fineweb2-ru": 2_348_137,
               "mc4-id": 151_645, "mc4-ru": 354_183,
               "hplt-id": 1_093_525, "taiga-ru": 0}
# Taiga-ru resume point for v16 (v14 manifest pq block, FINAL-REPORT §12):
# files 00000-00011 fully consumed (20,875 rows each); file 00012 at
# row 8,852 / 20,875. Seeded into the pq checkpoint on a fresh v16 run.
TAIGA_START = {
  "file_idx": 12,
  "row": 8852,
  "file_rows": {
    "data/train-00000-of-00083-5a836a36820bbc21.parquet": 20875,
    "data/train-00001-of-00083-6a059492052de562.parquet": 20875,
    "data/train-00002-of-00083-6ab99ef2eda1556f.parquet": 20875,
    "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet": 20875,
    "data/train-00004-of-00083-a2a0fa5d28e7d578.parquet": 20875,
    "data/train-00005-of-00083-806fed410a43fb46.parquet": 20875,
    "data/train-00006-of-00083-d6b9c39127b6005d.parquet": 20875,
    "data/train-00007-of-00083-206cf23f973bc8bf.parquet": 20875,
    "data/train-00008-of-00083-7798c30b513dd0ef.parquet": 20875,
    "data/train-00009-of-00083-c787d06e48840b95.parquet": 20875,
    "data/train-00010-of-00083-66b33c44bfe0c7b5.parquet": 20875,
    "data/train-00011-of-00083-78892dcef5e8b81a.parquet": 20875,
    "data/train-00012-of-00083-87b563f3f9bce3cf.parquet": 20875
  }
}
# Wikipedia-ru: EXHAUSTED (all 21 files read). Not used in v14.

# Stream watchdog: max seconds with no doc yielded before we abort.
STREAM_TIMEOUT_S = 600
HEARTBEAT_S = 120

_log_fh = None
_LAST_ACTIVITY = time.time()
_PHASE = ["startup"]


def _ts():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if _log_fh:
        try:
            _log_fh.write(line + "\n"); _log_fh.flush()
        except Exception:
            pass


def touch(phase=None):
    global _LAST_ACTIVITY
    _LAST_ACTIVITY = time.time()
    if phase:
        _PHASE[0] = phase


def _watchdog():
    while True:
        time.sleep(HEARTBEAT_S)
        idle = time.time() - _LAST_ACTIVITY
        log(f"[watchdog] alive phase={_PHASE[0]} idle={idle:.0f}s")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def is_cyrillic(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def clean_doc(text: str, lang: str):
    # IDENTICAL to v1-v10 — do not change filtering behaviour.
    text = text.replace("\xa0", " ")
    paras = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [re.sub(r"\s+", " ", text.strip())]
    out = []
    buf = ""
    for p in paras:
        if len(p) > 6000:
            start = 0
            while start < len(p):
                end = min(start + 5000, len(p))
                if end < len(p):
                    cut = p.rfind(". ", start, end)
                    if cut > start + 1000:
                        end = cut + 1
                out.append(p[start:end].strip())
                start = end
            continue
        if len(buf) + len(p) + 1 <= 6000 and buf:
            buf += " " + p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    kept = []
    for d in out:
        if not (200 <= len(d) <= 12000):
            continue
        letters = sum(c.isalpha() for c in d)
        if letters / len(d) < 0.60:
            continue
        digits = sum(c.isdigit() for c in d)
        if digits / len(d) > 0.20:
            continue
        lines = [l.strip() for l in d.split(". ") if l.strip()]
        if lines and len(set(lines)) / len(lines) < 0.70:
            continue
        cyr = sum(is_cyrillic(c) for c in d) / max(1, letters)
        if lang == "ru" and cyr < 0.30:
            continue
        if lang == "id" and cyr > 0.05:
            continue
        kept.append(d)
    return kept


# MANDATORY SPAM DENYLIST (user requirement 2026-10-26): HPLT samples
# contained gambling SEO spam that clean_doc() keeps. Applied to the
# RAW doc BEFORE clean_doc(); any match drops the whole doc. Counts
# are logged per source as spam_dropped in progress.json.
SPAM_DENYLIST = ["judi", "slot online", "situs judi", "togel", "gacor",
                 "casino", "taruhan", "maxwin", "rtp slot", "bandar judi",
                 "poker online",
                 # Extended 2026-10-26 after the filter test: the 2nd chunk
                 # of the known HPLT spam article ("situs slot deposit
                 # pulsa") contains none of the base terms, so these
                 # observed variants were added (research report: "extend
                 # if the first HPLT sample shows more variants").
                 "situs slot", "slot deposit", "deposit pulsa", "agen slot"]


def is_spam(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in SPAM_DENYLIST)


class ShardWriter:
    def __init__(self, lang):
        self.lang = lang
        self.idx = 0
        self.docs = 0
        self.bytes = 0
        self.f = None
        self._open_new()

    def _open_new(self):
        OUT.mkdir(parents=True, exist_ok=True)
        while (OUT / f"{self.lang}-{self.idx:05d}.jsonl").exists():
            self.idx += 1
        self.path = OUT / f"{self.lang}-{self.idx:05d}.jsonl"
        self.f = open(self.path, "w", encoding="utf-8")
        self.docs = 0; self.bytes = 0

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        b = line.encode("utf-8")
        if (self.docs >= SHARD_DOCS or self.bytes + len(b) > SHARD_BYTES) and self.docs:
            self.f.close(); self.idx += 1; self._open_new()
            line = json.dumps(rec, ensure_ascii=False) + "\n"; b = line.encode("utf-8")
        self.f.write(line); self.docs += 1; self.bytes += len(b)

    def close(self):
        if self.f: self.f.close(); self.f = None


def repair_shards():
    """A kill -9 can leave a torn last line in the newest shard of a
    language (v5 hit this: trailing NUL/truncated line). The packer
    would choke on it, so on start-up validate the last line of the
    newest shard per language and truncate a partial line. Only the
    newest shard can be torn — older shards were closed cleanly."""
    for lang in ("id", "ru"):
        shards = sorted(OUT.glob(f"{lang}-*.jsonl"))
        if not shards:
            continue
        f = shards[-1]
        try:
            data = f.read_bytes()
        except Exception:
            continue
        if not data:
            continue
        lines = data.split(b"\n")
        # If the file ends with "\n", the last element is b"" and the
        # last real line is lines[-2]; otherwise the last line is torn
        # unless it still parses as JSON.
        if data.endswith(b"\n"):
            last = lines[-2] if len(lines) >= 2 else b""
            try:
                json.loads(last.decode("utf-8"))
                continue  # clean
            except Exception:
                keep = data[: len(data) - len(last) - 1]
        else:
            try:
                json.loads(lines[-1].decode("utf-8"))
                continue
            except Exception:
                keep = data[: len(data) - len(lines[-1])]
        f.write_bytes(keep)
        log(f"[repair] truncated torn last line in {f.name} ({len(data) - len(keep)} bytes removed)")


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"sources": {}, "total": {"docs": 0, "chars": 0, "bytes": 0},
            "started": _ts()}


def save_progress(p):
    p["updated"] = _ts()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=2, ensure_ascii=False))


EXTRA_HASH_DIRS = [ROOT / "data_v13" / "text", ROOT / "data_v14" / "text"]


def _extend_with_extra_dirs(hashes):
    """GHA variant: no-op. The cache was already extended on the VM with
    v13+v14 text (extend_hash_v16.py) before upload; meta extra_done
    lists data_v13/data_v14. The extra dirs do not exist on the runner."""
    log("[hashes] GHA: cache pre-extended on VM, skipping extra-dir scan")
    return


def rebuild_hashes():
    """Hashes of all kept docs in v1..v10 (via a disk cache — rebuilding
    from ~20 GB of JSONL takes minutes on every resume and was one of
    v6's silent phases) plus v13+v14 (see _extend_with_extra_dirs) plus
    a fresh scan of OUT (this batch's own text, which grows during the
    run). The cache is only trusted when the recorded file inventory
    (path+size+mtime) still matches disk."""
    hashes = _load_or_build_prior_hashes()
    _extend_with_extra_dirs(hashes)
    if OUT.exists():
        t0 = time.time(); n = 0
        log(f"[hashes] scanning own output {OUT} ...")
        for f in sorted(OUT.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        t = json.loads(line)["text"]
                    except Exception:
                        continue
                    hashes.add(hashlib.sha256(norm(t).encode("utf-8")).hexdigest())
                    n += 1
        touch("hashes")
        log(f"[hashes] {OUT} done: {n} docs in {time.time()-t0:.0f}s, total_hashes={len(hashes)}")
    return hashes


def _prior_hash_dirs():
    return [ROOT / "data_full" / "text", ROOT / "data_v2" / "text",
            ROOT / "data_v3" / "text", ROOT / "data_v4" / "text",
            ROOT / "data_v5" / "text", ROOT / "data_v6" / "text",
            ROOT / "data_v7" / "text", ROOT / "data_v8" / "text",
            ROOT / "data_v9" / "text", ROOT / "data_v10" / "text"]


def _load_or_build_prior_hashes():
    """GHA variant: the cache file is pre-built and verified on the VM
    (v1-v14 hashes, 17,258,649 lines) and downloaded from Drive. No
    v1-v10 text exists on the runner, so the inventory check cannot run
    — and the stock version would DELETE the cache on mismatch and
    rebuild to an empty set. Trust the file; verify the line count in
    the workflow before launching the builder."""
    cache_path = OUT.parent / "hash_cache_v1_v10.txt"
    hashes = set()
    t0 = time.time()
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                hashes.add(line)
    touch("hashes")
    log(f"[hashes] loaded {len(hashes)} cached hashes in {time.time()-t0:.0f}s (GHA pre-extended cache, v1-v14)")
    return hashes


class StreamTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise StreamTimeout(f"no doc yielded for {STREAM_TIMEOUT_S}s")


def timed_next(it):
    """next(it) with a SIGALRM watchdog so a hung HTTP stream aborts
    instead of blocking forever (v6 had no timeout at all)."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(STREAM_TIMEOUT_S)
        try:
            return next(it)
        finally:
            signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old)


# ---------------- FineWeb-2 via direct parquet (fix 1) ----------------

def pq_list_files(repo, prefix):
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def pq_row_count_remote(repo, path):
    """Row count + row groups from the parquet FOOTER only (HTTP range
    reads, no full download). Returns (rows, groups) or (None, None).
    NOTE: trust_env=True is REQUIRED — this VM reaches the internet via
    an egress proxy from the environment, and aiohttp ignores proxy env
    vars by default (fails with SSL WRONG_VERSION_NUMBER)."""
    import fsspec
    import pyarrow.parquet as pq
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    try:
        with fsspec.open(url, mode="rb", client_kwargs={"trust_env": True}) as f:
            pf = pq.ParquetFile(f)
            return pf.metadata.num_rows, pf.metadata.num_row_groups
    except Exception as e:
        log(f"[fw2] footer read failed for {path}: {type(e).__name__}: {str(e)[:150]}")
        return None, None


def pq_open_remote(repo, path):
    """Open a remote parquet file for REMOTE row-group reads
    (pyarrow over fsspec HTTP range reads). No full download: bytes are
    fetched once, on demand, and a resume re-opens at the saved row
    group instead of re-fetching. Returns (ParquetFile, closer)."""
    import fsspec
    import pyarrow.parquet as pq
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    f = fsspec.open(url, mode="rb", client_kwargs={"trust_env": True}).open()
    pf = pq.ParquetFile(f)
    return pf, f


def pq_download(repo, path):
    from huggingface_hub import hf_hub_download
    PARQUET_CACHE.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo, filename=path,
                           repo_type="dataset", local_dir=str(PARQUET_CACHE),
                           local_dir_use_symlinks=False)


def main():
    global _log_fh
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--id-chars", type=int, default=2_600_000_000)
    ap.add_argument("--ru-chars", type=int, default=1_750_000_000)
    ap.add_argument("--only", default=None,
                    help="comma-separated source keys to run, e.g. fineweb2-id")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default data_v16/text); for dry runs")
    args = ap.parse_args()

    global OUT, PROGRESS, LOGFILE, PARQUET_CACHE
    if args.out_dir:
        OUT = Path(args.out_dir) / "text"
        PROGRESS = Path(args.out_dir) / "progress.json"
        LOGFILE = Path(args.out_dir) / "build_text_v16.log"
        PARQUET_CACHE = Path(args.out_dir) / "cache_parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOGFILE, "a", encoding="utf-8")
    repair_shards()

    only = set(args.only.split(",")) if args.only else None
    threading.Thread(target=_watchdog, daemon=True).start()

    from datasets import load_dataset

    targets = {"id": args.id_chars, "ru": args.ru_chars}
    progress = load_progress() if args.resume else {
        "sources": {}, "total": {"docs": 0, "chars": 0, "bytes": 0},
        "started": _ts()}
    if args.resume:
        for _k, _v in progress["sources"].items():
            _lang = _k.rsplit("-", 1)[-1]
            _kept = sum(v.get("kept_chars", 0) for k, v in progress["sources"].items() if k.endswith(f"-{_lang}"))
            if _kept < targets[_lang]:
                _v["done"] = False
    touch("hashes")
    seen = rebuild_hashes()
    log(f"[build] resume={args.resume} seen_hashes={len(seen)} targets={targets} only={only}")
    writers = {}

    def writer(lang):
        if lang not in writers:
            writers[lang] = ShardWriter(lang)
        return writers[lang]

    def process_text(key, lang, text, st):
        if is_spam(text):
            st["spam_dropped"] = st.get("spam_dropped", 0) + 1
            return
        for doc in clean_doc(text, lang):
            h = hashlib.sha256(norm(doc).encode("utf-8")).hexdigest()
            if h in seen:
                st["duplicates"] += 1; continue
            seen.add(h)
            rec = {"id": f"{key}-{st['processed']}", "lang": lang, "source": key.split("-")[0], "text": doc}
            writer(lang).write(rec)
            st["kept_docs"] += 1; st["kept_chars"] += len(doc)
            progress["total"]["docs"] += 1; progress["total"]["chars"] += len(doc)
        touch(f"process:{key}")

    def kept_lang(lang):
        return sum(v.get("kept_chars", 0) for k, v in progress["sources"].items() if k.endswith(f"-{lang}"))

    failed = False
    for lang, src, kind, repo, cfg, cap, prefix in SOURCES:
        key = f"{src}-{lang}"
        if only and key not in only:
            continue
        st = progress["sources"].setdefault(key, {"processed": 0, "kept_docs": 0, "kept_chars": 0, "duplicates": 0, "filtered": 0, "spam_dropped": 0, "done": False})
        if st.get("done"):
            log(f"[build] skip {key} (done)"); continue
        if kept_lang(lang) >= targets[lang]:
            log(f"[build] {lang} target reached ({kept_lang(lang)} chars), skip {key}")
            st["done"] = True; save_progress(progress); continue
        log(f"[build] source {key} kind={kind} repo={repo} cfg={cfg} cap={cap} processed_so_far={st['processed']} kept_lang={kept_lang(lang)}/{targets[lang]}")
        t0 = time.time(); last_save = time.time()
        finished = False  # only True on genuine finish (fix 4)
        try:
            if kind == "parquet_local":
                files = [fp for fp in sorted(CACHE.glob("20231101.ru-*.parquet"))
                         if int(fp.name.split("-")[2]) >= WIKI_RU_START]
                done_files = set(st.get("files_done", []))
                log(f"[build] local parquet {key}: {len(files)} files, {len(done_files)} already done")
                for fp in files:
                    if fp.name in done_files:
                        continue
                    touch(f"parquet:{key}")
                    log(f"[build] parquet file {fp.name}")
                    ds = load_dataset("parquet", data_files=[str(fp)], split="train")
                    for ex in ds:
                        st["processed"] += 1
                        text = ex.get("text") or ex.get("content") or ""
                        if not text:
                            st["filtered"] += 1; continue
                        process_text(key, lang, text, st)
                        if kept_lang(lang) >= targets[lang]:
                            break
                        if cap and st["kept_chars"] >= cap:
                            log(f"[build] {key} cap reached ({st['kept_chars']} chars)"); break
                    done_files.add(fp.name)
                    st["files_done"] = sorted(done_files)
                    save_progress(progress)
                    if kept_lang(lang) >= targets[lang]:
                        log(f"[build] {lang} target reached in {key}"); break
                    if cap and st["kept_chars"] >= cap:
                        log(f"[build] {key} cap reached, stop parquet loop"); break
                finished = True
            elif kind == "parquet_remote":
                # --- remote parquet, direct row-group reads (fix 1) ---
                import pyarrow.parquet as pq
                fw = st.setdefault("pq", {"file_idx": 0, "row": 0,
                                          "file_rows": {}, "prefix_remaining": None})
                if (key == "taiga-ru" and fw["file_idx"] == 0 and fw["row"] == 0
                        and not fw["file_rows"] and fw["prefix_remaining"] is None):
                    # Fresh v16 run: resume Taiga where v14 stopped (§12).
                    fw.update({"file_idx": TAIGA_START["file_idx"],
                               "row": TAIGA_START["row"],
                               "file_rows": dict(TAIGA_START["file_rows"]),
                               "prefix_remaining": 0})
                    log(f"[build] pq {key}: resuming from v14 end state file_idx={fw['file_idx']} row={fw['row']}")
                if fw["prefix_remaining"] is None:
                    fw["prefix_remaining"] = STREAM_SKIP.get(key, 0)
                files = pq_list_files(repo, prefix)
                log(f"[build] pq {key}: repo={repo} {len(files)} parquet files, start file_idx={fw['file_idx']} row={fw['row']} prefix_remaining={fw['prefix_remaining']}")
                # Walk the v1-v10 consumed prefix by footer row counts only.
                while fw["prefix_remaining"] and fw["file_idx"] < len(files):
                    fname = files[fw["file_idx"]]
                    rows = fw["file_rows"].get(fname)
                    if rows is None:
                        rows, _g = pq_row_count_remote(repo, fname)
                        if rows is None:
                            # Footer read failed: fall back to downloading
                            # the file and reading its footer locally.
                            local = pq_download(repo, fname)
                            pf0 = pq.ParquetFile(local)
                            rows = pf0.metadata.num_rows
                        fw["file_rows"][fname] = rows
                        save_progress(progress)
                    if fw["prefix_remaining"] >= rows:
                        fw["prefix_remaining"] -= rows
                        fw["file_idx"] += 1
                        fw["row"] = 0
                        log(f"[build] pq {key}: skipped whole file {fname} ({rows} rows), prefix_remaining={fw['prefix_remaining']}")
                    else:
                        fw["row"] = fw["prefix_remaining"]
                        fw["prefix_remaining"] = 0
                        save_progress(progress)
                        log(f"[build] pq {key}: start mid-file {fname} at row {fw['row']}")
                        break
                # Process files from the checkpointed position.
                # Remote row-group reads (fix 1): bytes are fetched once,
                # on demand; a resume re-opens at the saved row group.
                # A failed remote open falls back to a resumable download
                # of that one file; a failed row-group read reopens once.
                def open_pf(fname):
                    try:
                        pf, fo = pq_open_remote(repo, fname)
                        return pf, fo
                    except Exception as e:
                        log(f"[build] pq {key}: remote open failed ({type(e).__name__}: {str(e)[:150]}), falling back to download")
                        local = pq_download(repo, fname)
                        return pq.ParquetFile(local), None

                while fw["file_idx"] < len(files):
                    fname = files[fw["file_idx"]]
                    touch(f"pq:{key}")
                    log(f"[build] pq {key}: opening {fname} (remote row-group reads)")
                    pf, closer = open_pf(fname)
                    rows = pf.metadata.num_rows
                    fw["file_rows"][fname] = rows
                    ngroups = pf.metadata.num_row_groups
                    start_row = fw["row"]
                    # Find the row group containing start_row.
                    gi = 0; acc = 0
                    while gi < ngroups and acc + pf.metadata.row_group(gi).num_rows <= start_row:
                        acc += pf.metadata.row_group(gi).num_rows; gi += 1
                    offset_in_group = start_row - acc
                    log(f"[build] pq {key}: {fname} rows={rows} groups={ngroups} resume group={gi} offset={offset_in_group}")
                    stop = False
                    for g in range(gi, ngroups):
                        try:
                            table = pf.read_row_group(g, columns=["text"])
                        except Exception as e:
                            log(f"[build] pq {key}: row-group {g} read failed ({type(e).__name__}), reopening once")
                            try:
                                if closer is not None:
                                    closer.close()
                            except Exception:
                                pass
                            pf, closer = open_pf(fname)
                            table = pf.read_row_group(g, columns=["text"])
                        texts = table.column("text").to_pylist()
                        for j, text in enumerate(texts):
                            if g == gi and j < offset_in_group:
                                continue
                            st["processed"] += 1
                            fw["row"] += 1
                            if not text:
                                st["filtered"] += 1; continue
                            process_text(key, lang, text, st)
                            if kept_lang(lang) >= targets[lang]:
                                stop = True; break
                            if cap and st["kept_chars"] >= cap:
                                stop = True; break
                        # Checkpoint after EVERY row group (fix 1): a kill
                        # here costs at most one row group, re-deduped.
                        save_progress(progress)
                        if time.time() - last_save > 60:
                            last_save = time.time()
                            log(f"[build] {key} file={fname} group={g}/{ngroups} processed={st['processed']} kept={st['kept_docs']} chars={st['kept_chars']} total_chars={progress['total']['chars']} elapsed={time.time()-t0:.0f}s")
                        if stop:
                            break
                    try:
                        if closer is not None:
                            closer.close()
                    except Exception:
                        pass
                    if stop:
                        finished = (kept_lang(lang) >= targets[lang]) or (cap and st["kept_chars"] >= cap)
                        save_progress(progress)
                        break
                    fw["file_idx"] += 1
                    fw["row"] = 0
                    save_progress(progress)
                    log(f"[build] pq {key}: finished file {fname}, next file_idx={fw['file_idx']}")
                if fw["file_idx"] >= len(files):
                    finished = True
                    log(f"[build] pq {key}: all train files exhausted")
            else:  # streaming (mC4 / HPLT) — small skip, but with heartbeat + watchdog
                if src == "hplt":
                    log(f"[build] hplt {key}: streaming startup may take a few minutes (shard dirs list 140-475 files each) — not a stall")
                ds = load_dataset(repo, cfg, split="train", streaming=True)
                it = iter(ds)
                skip = STREAM_SKIP.get(key, 0) + st["processed"]
                if skip:
                    log(f"[build] skipping {skip} docs in {key} (resume/consumed prefix)")
                    skipped = 0; t_skip = time.time()
                    while skipped < skip:
                        try:
                            timed_next(it)
                        except StopIteration:
                            break
                        skipped += 1
                        if skipped % 5000 == 0:
                            touch(f"skip:{key}")
                            save_progress(progress)
                            log(f"[build] skip {key}: {skipped}/{skip} ({skipped/(time.time()-t_skip):.0f} docs/s)")
                    touch(f"skip:{key}")
                    log(f"[build] skip {key} done: {skipped} docs in {time.time()-t_skip:.0f}s")
                for ex in timed_next_iter(it):
                    st["processed"] += 1
                    text = ex.get("text") or ex.get("content") or ""
                    if not text:
                        st["filtered"] += 1; continue
                    process_text(key, lang, text, st)
                    if time.time() - last_save > 120:
                        save_progress(progress); last_save = time.time()
                        log(f"[build] {key} processed={st['processed']} kept={st['kept_docs']} chars={st['kept_chars']} total_chars={progress['total']['chars']} elapsed={time.time()-t0:.0f}s")
                    if cap and st["kept_chars"] >= cap:
                        log(f"[build] {key} cap reached ({st['kept_chars']} chars)"); finished = True; break
                    if kept_lang(lang) >= targets[lang]:
                        log(f"[build] {lang} target reached in {key}"); finished = True; break
                else:
                    finished = True  # iterator exhausted
        except StopIteration:
            log(f"[build] {key} exhausted"); finished = True
        except Exception as e:
            # fix 4: do NOT mark done; save and fail loudly for retry.
            log(f"[build] ERROR in {key}: {type(e).__name__}: {e}")
            st["error"] = str(e)[:300]
            save_progress(progress)
            failed = True
            break
        if finished:
            st["done"] = True
        save_progress(progress)
        log(f"[build] done {key}: {st}")
        if failed:
            break
    for w in writers.values():
        w.close()
    save_progress(progress)
    log("[build] FINAL " + json.dumps(progress, indent=2, ensure_ascii=False))
    if _log_fh:
        _log_fh.close()
    if failed:
        log("[build] FAILED — progress saved, done=False, retry with --resume")
        sys.exit(2)


def timed_next_iter(it):
    """Yield from a streaming iterator with the SIGALRM watchdog on
    every read (v6's plain `for ex in it` could hang forever)."""
    while True:
        try:
            yield timed_next(it)
        except StopIteration:
            return


if __name__ == "__main__":
    main()
