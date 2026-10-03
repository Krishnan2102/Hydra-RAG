import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset

SEED = 42                  # same seed = same eval queries every run
N_EVAL_QUERIES = 100
TARGET_PASSAGES = 100_000
OUT_DIR = Path("data")

def clean_query(q: str) -> str:
    """Strip leading punctuation like '. what is...' """
    return re.sub(r"^[\s\.\?\!,;:]+", "", q).strip()


def is_usable(row) -> bool:
    """Keep rows with a real reference answer AND at least one relevant passage."""
    answers = [a.strip() for a in row["answers"] if a.strip()]
    if not answers or answers[0].lower().startswith("no answer present"):
        return False
    return 1 in row["passages"]["is_selected"]


class PassageStore:
    """Collects unique passages and gives each one an integer ID."""

    def __init__(self):
        self.passages = []
        self._ids = {}   # text -> id, used to remove duplicates

    def add(self, text: str, url: str) -> int:
        text = text.strip()
        if text not in self._ids:
            self._ids[text] = len(self.passages)
            self.passages.append({
                "id": len(self.passages),
                "text": text,
                "source": urlparse(url).netloc or "msmarco",
            })
        return self._ids[text]

    def __len__(self):
        return len(self.passages)

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    usable = [i for i, row in enumerate(ds) if is_usable(row)]
    eval_idx = random.sample(usable, N_EVAL_QUERIES)

    store = PassageStore()
    eval_set = []
    for i in eval_idx:
        row = ds[i]
        p = row["passages"]
        relevant_ids = []
        for flag, text, url in zip(p["is_selected"], p["passage_text"], p["url"]):
            pid = store.add(text, url)
            if flag == 1:
                relevant_ids.append(pid)
        eval_set.append({
            "query_id": row["query_id"],
            "query": clean_query(row["query"]),
            "reference": row["answers"][0].strip(),
            "relevant_ids": relevant_ids,
        })

        eval_set_idx = set(eval_idx)
    others = [i for i in range(len(ds)) if i not in eval_set_idx]
    random.shuffle(others)
    for i in others:
        if len(store) >= TARGET_PASSAGES:
            break
        p = ds[i]["passages"]
        for text, url in zip(p["passage_text"], p["url"]):
            store.add(text, url)

        all_ids = {p["id"] for p in store.passages}
    assert all(set(e["relevant_ids"]) <= all_ids for e in eval_set), "missing relevant passage!"

    with open(OUT_DIR / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in store.passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"Passages: {len(store)}")
    print(f"Eval queries: {len(eval_set)}")
    print(f"Avg relevant per query: {sum(len(e['relevant_ids']) for e in eval_set)/len(eval_set):.2f}")
    print("Example:", eval_set[0]["query"], "->", eval_set[0]["reference"][:80])


if __name__ == "__main__":
    main() 