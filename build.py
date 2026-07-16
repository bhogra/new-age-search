"""Export the keyword-searcher index as static files for the website.

Run once: python build.py --src /path/to/keyword-searcher/index --out data
Needs: numpy, pandas, pyarrow.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_SHARDS = 4

# Fold typographic characters to ASCII so the browser can hold the corpus as a
# compact 1-byte string (any char > U+00FF would double its memory).
FOLD = str.maketrans({"—": "--", "–": "-", "“": '"', "”": '"',
                      "‘": "'", "’": "'", "œ": "oe", "Œ": "Oe"})


def normalize(s):
    s = s.translate(FOLD)
    return s if s.isascii() else "".join(c if ord(c) < 256 else "?" for c in s)


def main(src, out):
    out.mkdir(exist_ok=True)
    chunks = pd.read_parquet(src / "chunks.parquet")
    n = len(chunks)
    per = -(-n // N_SHARDS)  # ceil
    sizes = {}

    # --- embeddings: per-vector symmetric int8 + float32 scales ---
    emb = np.load(src / "embeddings.npy").astype(np.float32)
    scales = np.abs(emb).max(axis=1) / 127.0
    q = np.round(emb / scales[:, None]).astype(np.int8)
    for i in range(N_SHARDS):
        f = out / f"emb_{i}.bin"
        f.write_bytes(q[i * per:(i + 1) * per].tobytes())
        sizes[f.name] = f.stat().st_size
    (out / "emb_scales.bin").write_bytes(scales.astype("<f4").tobytes())
    sizes["emb_scales.bin"] = (out / "emb_scales.bin").stat().st_size

    # --- passage text: \x1f-joined, pre-gzipped shards ---
    texts = [normalize(t) for t in chunks["text"]]
    assert not any("\x1f" in t for t in texts)
    folded = sum(t != o for t, o in zip(texts, chunks["text"]))
    print(f"normalized text in {folded:,} of {n:,} passages")
    for i in range(N_SHARDS):
        f = out / f"text_{i}.gz"
        blob = "\x1f".join(texts[i * per:(i + 1) * per]).encode("utf-8")
        with gzip.GzipFile(f, "wb", compresslevel=9, mtime=0) as g:
            g.write(blob)
        sizes[f.name] = f.stat().st_size

    # --- vocab ---
    with gzip.GzipFile(out / "vocab.txt.gz", "wb", compresslevel=9, mtime=0) as g:
        g.write((src / "vocab.txt").read_bytes())
    sizes["vocab.txt.gz"] = (out / "vocab.txt.gz").stat().st_size

    # --- article table: chunks are contiguous per (issue_id, article_num) ---
    key = chunks["issue_id"] + "|" + chunks["article_num"].astype(str)
    art_start = np.flatnonzero(key != key.shift()).tolist()
    starts = np.array(art_start + [n])
    art_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    assert (chunks["para_idx"].values == np.arange(n) - starts[art_of]).all()
    assert chunks.groupby(key, sort=False)["heading"].nunique().eq(1).all()

    issues = sorted(chunks["issue_id"].unique())
    dates = pd.read_csv(src / "issue_dates.csv").set_index("issue_id")["date"]
    assert all(i in dates.index for i in issues)
    headings = sorted({normalize(h) for h in chunks["heading"]})
    issue_ix = {s: i for i, s in enumerate(issues)}
    head_ix = {s: i for i, s in enumerate(headings)}
    arts = chunks.iloc[art_start]
    meta = {
        "n_chunks": n, "rows_per_shard": per, "bytes": sizes,
        "issues": issues, "issue_dates": [dates[i] for i in issues],
        "headings": headings,
        "art_start": art_start,
        "art_issue": [issue_ix[s] for s in arts["issue_id"]],
        "art_head": [head_ix[normalize(s)] for s in arts["heading"]],
        "art_num": arts["article_num"].astype(int).tolist(),
    }
    with gzip.GzipFile(out / "meta.json.gz", "wb", compresslevel=9, mtime=0) as g:
        g.write(json.dumps(meta, separators=(",", ":")).encode("utf-8"))
    print(f"{n:,} passages, {len(art_start):,} articles, {len(issues)} issues")
    for name, size in sizes.items():
        print(f"  {name}: {size / 1e6:.1f} MB")
    print(f"  meta.json.gz: {(out / 'meta.json.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data"))
    a = p.parse_args()
    main(a.src, a.out)
