"""Check that the exported int8 index ranks passages like the original float index.

Run: python compare.py --src /path/to/keyword-searcher/index [--query "..."]
Needs: numpy, sentence-transformers.
"""
import argparse
from pathlib import Path

import numpy as np

QUERIES = ["suffragette", "guild socialism", "the education of women"]
PREFIX = "Represent this sentence for searching relevant passages: "


def main(src, queries):
    from sentence_transformers import SentenceTransformer
    emb = np.load(src / "embeddings.npy").astype(np.float32)
    data = Path(__file__).parent / "data"
    scales = np.frombuffer((data / "emb_scales.bin").read_bytes(), dtype="<f4")
    q8 = np.frombuffer(b"".join((data / f"emb_{i}.bin").read_bytes() for i in range(4)),
                       dtype=np.int8).reshape(len(emb), 384)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    for query in queries:
        v = model.encode([PREFIX + query], normalize_embeddings=True)[0]
        top_f = np.argsort(-(emb @ v))[:10]
        top_q = np.argsort(-((q8 @ v) * scales))[:10]
        overlap = len(set(top_f) & set(top_q))
        print(f"{query!r}: top-10 overlap {overlap}/10")
        print(f"  float32: {top_f.tolist()}")
        print(f"  int8:    {top_q.tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--query", action="append")
    a = p.parse_args()
    main(a.src, a.query or QUERIES)
