import itertools
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall
from ragas.run_config import RunConfig

from retrieval.retriever import HydraRetriever

# -----------------------------------------------------------------------------
# Explicit .env loading from project root
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


# =====================================================================
# 1. Multi-Account Rotating ChatGroq Wrapper
# =====================================================================
class RotatingChatGroq(ChatGroq):
    """
    Cycles across distinct Groq API keys in round-robin fashion,
    ensuring each sequential request consumes quota from a different account.
    """
    _key_cycle: itertools.cycle
    _lock: threading.Lock

    def __init__(self, key_list: List[str], **kwargs):
        if not key_list:
            raise ValueError("key_list cannot be empty.")
        super().__init__(groq_api_key=key_list[0], **kwargs)
        self._key_cycle = itertools.cycle(key_list)
        self._lock = threading.Lock()

    def _get_next_key(self) -> str:
        with self._lock:
            return next(self._key_cycle)

    def invoke(self, *args, **kwargs):
        self.groq_api_key = self._get_next_key()
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        self.groq_api_key = self._get_next_key()
        return await super().ainvoke(*args, **kwargs)


# =====================================================================
# 2. Metric Parsing Helpers
# =====================================================================
def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_metric_mean(result_obj: Any, metric_name: str) -> float:
    val = result_obj[metric_name]
    if isinstance(val, (list, np.ndarray)):
        clean_vals = [float(v) for v in val if v is not None and not np.isnan(v)]
        return round(float(np.mean(clean_vals)), 4) if clean_vals else 0.0
    return round(float(val), 4)


def extract_per_query_list(result_obj: Any, metric_name: str) -> List[float]:
    val = result_obj[metric_name]
    if isinstance(val, (list, np.ndarray)):
        return [round(float(v), 4) if (v is not None and not np.isnan(v)) else 0.0 for v in val]
    return [round(float(val), 4)]


# =====================================================================
# 3. Main Evaluation Pipeline
# =====================================================================
def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "dense"
    if mode not in ["dense", "hybrid"]:
        print(f"Unknown mode '{mode}'. Defaulting to 'dense'. (Allowed: dense, hybrid)")
        mode = "dense"

    # Support either GROQ_API_KEYS="key1,key2,key3" or single GROQ_API_KEY="key"
    raw_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
    key_pool = [k.strip().strip("'\"") for k in raw_keys.split(",") if k.strip()]

    if not key_pool:
        raise ValueError(
            "No Groq API keys found. Ensure GROQ_API_KEYS is defined in your .env file."
        )

    print(f"Loaded {len(key_pool)} rotating API key(s) for evaluation.")

    cfg = load_config()
    paths = cfg["paths"]

    eval_path = Path(paths["eval_set"])
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # 20 queries strictly fulfills FR-2 (>= 20 queries)
    num_eval_queries = cfg.get("evaluation", {}).get("sample_size", 20)
    judge_model = "openai/gpt-oss-120b"

    print(f"Loading queries from {eval_path}...")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    eval_subset = eval_data[:num_eval_queries]
    print(f"Evaluating {len(eval_subset)} queries on Groq ({judge_model}) | Mode: {mode.upper()}")

    retriever = HydraRetriever()

    user_inputs = []
    retrieved_contexts = []
    references = []
    retrieved_doc_ids = []
    gold_hit_count = 0

    print(f"Retrieving top-5 candidates over Qdrant via {mode} search...")
    for item in eval_subset:
        q = item["query"]
        gold_ids = set(item.get("relevant_ids", []))

        if mode == "dense":
            res = retriever.search_dense(q, limit=5)
        else:
            res = retriever.search_hybrid(q, limit=5)

        hits = res.get("hits", [])
        top_ids = [hit["id"] for hit in hits]

        # Context truncation to 400 characters protects the TPM allocation
        contexts = [hit["text"][:400] for hit in hits]

        if any(gid in top_ids for gid in gold_ids):
            gold_hit_count += 1

        user_inputs.append(q)
        retrieved_contexts.append(contexts)
        references.append(item.get("reference", ""))
        retrieved_doc_ids.append(top_ids)

    hit_rate = gold_hit_count / len(eval_subset)
    print(f"Ground Truth Label Hit Rate (Recall@5): {hit_rate:.4f} ({gold_hit_count}/{len(eval_subset)})")

    ragas_dataset = Dataset.from_dict({
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "reference": references
    })

    print(f"Initializing rotating ChatGroq judge with {judge_model}...")
    base_llm = RotatingChatGroq(
        key_list=key_pool,
        model=judge_model,
        temperature=0.0,
        request_timeout=60.0,
        max_retries=8
    )
    evaluator_llm = LangchainLLMWrapper(base_llm)

    # Attach local FastEmbed BGE-small embeddings to prevent OpenAI checks
    print("Binding local FastEmbed BGE-small embeddings...")
    local_embed = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    evaluator_embeddings = LangchainEmbeddingsWrapper(local_embed)

    # Serial processing: max_workers=1 prevents burst concurrency rate limits
    run_config = RunConfig(
        max_workers=1,
        max_retries=8,
        max_wait=45,
        timeout=120
    )

    print("Executing RAGAS evaluation (strictly serial, max_workers=1)...")
    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=[context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    mean_precision = get_metric_mean(eval_result, "context_precision")
    mean_recall = get_metric_mean(eval_result, "context_recall")
    per_precision = extract_per_query_list(eval_result, "context_precision")
    per_recall = extract_per_query_list(eval_result, "context_recall")

    output_payload = {
        "phase": f"Phase {'1: Dense Baseline' if mode == 'dense' else '2: Hybrid Search'}",
        "mode": mode,
        "sample_size": len(eval_subset),
        "judge_model": f"groq/{judge_model}",
        "accounts_rotated": len(key_pool),
        "scores": {
            "context_precision": mean_precision,
            "context_recall": mean_recall,
            "label_recall_at_5": round(hit_rate, 4)
        },
        "query_evaluations": [
            {
                "query": q,
                "gold_ids": item.get("relevant_ids", []),
                "retrieved_ids": r_ids,
                "context_precision": p,
                "context_recall": r
            }
            for q, item, r_ids, p, r in zip(
                user_inputs, eval_subset, retrieved_doc_ids, per_precision, per_recall
            )
        ]
    }

    out_file = results_dir / f"ragas_{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  RAGAS EVALUATION COMPLETE: {mode.upper()} (Groq Free Tier)")
    print("=" * 60)
    print(f"Context Precision:  {mean_precision:.4f}")
    print(f"Context Recall:     {mean_recall:.4f}")
    print(f"Label Recall@5:     {hit_rate:.4f}")
    print("=" * 60)
    print(f"Full report saved to: {out_file}")


if __name__ == "__main__":
    main()