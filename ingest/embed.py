import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 128
MAX_SEQ_LEN = 256


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    passages_file = DATA_DIR / "passages.jsonl"
    output_file = DATA_DIR / ("embeddings_sample.npy" if limit else "embeddings.npy")

    if not passages_file.exists():
        raise FileNotFoundError(f"{passages_file} missing. Run ingest/prepare_data.py first.")

    with open(passages_file, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if limit:
        texts = texts[:limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | passages: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LEN

    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Done in {(time.perf_counter() - start) / 60:.1f} min | shape {embeddings.shape}")

    np.save(output_file, embeddings.astype("float32"))
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()