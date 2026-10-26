#!/usr/bin/env python3
"""Tokenize trixtom v16 text and pack into 8192-token sequences.

Uses the EXISTING v1 tokenizer (vocab 32000); no retraining for v16.

- Loads data_full/tokenizer/tokenizer.json (vocab 32000).
- Docs are read from text shards interleaved id/ru, shuffled in 20k-doc
  blocks (seed 42) so packed sequences mix languages.
- Each doc is encoded, <eos> appended; tokens are concatenated and cut
  into exact 8192-token sequences, saved as uint16 little-endian .bin.
- Shards are capped at 512 MiB (268,435,456 tokens) for reliable Drive
  upload. Progress is written to data_full/pack_progress.json after
  each shard; a final manifest.json is written at the end.
"""
import hashlib, json, random, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data_v16" / "text"
TOK = ROOT / "data_full" / "tokenizer" / "tokenizer.json"
OUT = ROOT / "data_v16" / "packed"
SEQ_LEN = 8192
SHARD_MAX_TOKENS = 268_435_456  # 512 MiB at 2 bytes/token
BLOCK = 20000

def iter_docs():
    """Yield (lang, source, text) interleaving id/ru shards in order."""
    id_files = sorted(TEXT.glob("id-*.jsonl"))
    ru_files = sorted(TEXT.glob("ru-*.jsonl"))
    for i in range(max(len(id_files), len(ru_files))):
        for files, lang in [(id_files, "id"), (ru_files, "ru")]:
            if i < len(files):
                with open(files[i], encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            yield lang, rec.get("source", "unknown"), rec["text"]
                        except Exception:
                            continue

def main():
    tok = Tokenizer.from_file(str(TOK))
    eos = tok.token_to_id("<eos>")
    print(f"[pack] eos_id={eos} vocab={tok.get_vocab_size()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    shard_idx = 0
    shard_tokens = 0
    shard_path = OUT / f"shard-{shard_idx:05d}.bin"
    shard_f = open(shard_path, "wb")
    buf = []  # list of ints, flushed in 8192-token chunks
    total_tokens = 0
    total_seqs = 0
    docs = 0
    lang_tokens = {"id": 0, "ru": 0}
    lang_docs = {"id": 0, "ru": 0}
    source_tokens = {}
    source_docs = {}
    shard_stats = []
    t0 = time.time()
    block = []

    def flush_chunks(final=False):
        nonlocal shard_tokens, shard_idx, shard_f, total_tokens, total_seqs, shard_path
        while len(buf) >= SEQ_LEN or (final and buf):
            if len(buf) < SEQ_LEN:
                break
            seq = np.array(buf[:SEQ_LEN], dtype="<u2")
            if shard_tokens + SEQ_LEN > SHARD_MAX_TOKENS:
                shard_f.close()
                shard_stats.append({"file": shard_path.name, "tokens": shard_tokens, "sequences": shard_tokens // SEQ_LEN})
                shard_idx += 1
                shard_path = OUT / f"shard-{shard_idx:05d}.bin"
                shard_f = open(shard_path, "wb")
                shard_tokens = 0
                print(f"[pack] new shard {shard_path.name}", flush=True)
            shard_f.write(seq.tobytes())
            del buf[:SEQ_LEN]
            shard_tokens += SEQ_LEN
            total_tokens += SEQ_LEN
            total_seqs += 1
        if final and buf:
            # pad last partial sequence with <pad> (id 0) to a full 8192
            pad = tok.token_to_id("<pad>")
            seq = buf + [pad] * (SEQ_LEN - len(buf))
            arr = np.array(seq, dtype="<u2")
            shard_f.write(arr.tobytes())
            shard_tokens += SEQ_LEN
            total_tokens += SEQ_LEN
            total_seqs += 1
            buf.clear()

    def encode_block(block):
        nonlocal docs
        texts = [t for _, _, t in block]
        encs = tok.encode_batch(texts)
        for (lang, src, _), enc in zip(block, encs):
            ids = enc.ids + [eos]
            lang_tokens[lang] += len(ids)
            lang_docs[lang] += 1
            source_tokens[src] = source_tokens.get(src, 0) + len(ids)
            source_docs[src] = source_docs.get(src, 0) + 1
            docs += 1
            buf.extend(ids)
            flush_chunks()

    for lang, src, text in iter_docs():
        block.append((lang, src, text))
        if len(block) >= BLOCK:
            rng.shuffle(block)
            encode_block(block)
            block = []
            print(f"[pack] docs={docs} tokens={total_tokens} seqs={total_seqs} elapsed={time.time()-t0:.0f}s", flush=True)
    if block:
        rng.shuffle(block)
        encode_block(block)
    flush_chunks(final=True)
    shard_f.close()
    shard_stats.append({"file": shard_path.name, "tokens": shard_tokens, "sequences": shard_tokens // SEQ_LEN})
    # hashes + sizes
    for s in shard_stats:
        p = OUT / s["file"]
        s["bytes"] = p.stat().st_size
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        s["sha256"] = h.hexdigest()
    manifest = {
        "project": "trixtom-1.0-magorch",
        "version": "v16",
        "seq_len": SEQ_LEN,
        "dtype": "uint16 little-endian",
        "total_tokens": total_tokens,
        "total_sequences": total_seqs,
        "total_docs": docs,
        "tokens_by_lang": lang_tokens,
        "docs_by_lang": lang_docs,
        "tokenizer": "data_full/tokenizer/tokenizer.json",
        "tokenizer_note": "v1 tokenizer reused for v16; no retraining",
        "dedup_against": "v1..v14 (extended disk hash cache) + v15 (pre-pack cross-check script dedupe_v16_vs_v15.py dropped cross-batch duplicates before packing)",
        "tokens_by_source": source_tokens,
        "docs_by_source": source_docs,
        "tokenizer_vocab_size": tok.get_vocab_size(),
        "shards": shard_stats,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    # merge build progress (per-source processed/kept chars) + cumulative totals
    prog_path = ROOT / "data_v16" / "progress.json"
    if prog_path.exists():
        manifest["sources"] = json.loads(prog_path.read_text()).get("sources", {})
    cum = total_tokens
    for mf in ("data_full/manifest.json", "data_v2/manifest.json", "data_v3/manifest.json", "data_v4/manifest.json", "data_v5/manifest.json", "data_v6/manifest.json", "data_v7/manifest.json", "data_v8/manifest.json", "data_v9/manifest.json", "data_v10/manifest.json", "data_v11/manifest.json", "data_v12/manifest.json", "data_v13/manifest.json", "data_v14/manifest.json", "data_v15/manifest.json"):
        mp = ROOT / mf
        if mp.exists():
            cum += json.loads(mp.read_text()).get("total_tokens", 0)
    manifest["cumulative_tokens_v1_v2_v3_v4_v5_v6_v7_v8_v9_v10_v11_v12_v13_v14_v15_v16"] = cum
    manifest["remaining_to_15B_tokens"] = 15_000_000_000 - cum
    (ROOT / "data_v16" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[pack] FINAL", json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
