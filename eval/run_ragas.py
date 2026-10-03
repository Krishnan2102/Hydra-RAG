import os
import sys
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from datasets import Dataset
from retrieval.dense import search
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas.llms import LangchainLLMWrapper

load_dotenv()

def calculate_judge_free_metrics(retrieved_ids, relevant_ids):
    intersection = set(retrieved_ids).intersection(set(relevant_ids))
    hit = 1 if intersection else 0
    
    mrr = 0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            mrr = 1.0 / rank
            break
            
    precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0
    recall = len(intersection) / len(relevant_ids) if relevant_ids else 0
    
    return {"hit": hit, "mrr": mrr, "precision": precision, "recall": recall}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of queries to evaluate (useful for testing rate limits)")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for Groq LLM processing")
    args = parser.parse_args()

    eval_file = Path("data/eval_set.json")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "phase1_ragas.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)[:args.limit]

    print(f"Starting evaluation on {len(eval_queries)} queries...")
    
    # Prepare data for RAGAS
    ragas_data = {
        "user_input": [],
        "retrieved_contexts": [],
        "reference": []
    }
    
    judge_free_totals = {"hit": 0, "mrr": 0, "precision": 0, "recall": 0}

    for i, q_data in enumerate(eval_queries):
        query = q_data["query"]
        relevant_ids = q_data["relevant_ids"]
        
        # Retrieve top 5 using our dense retriever
        results = search(query, mode="dense", top_k=5)
        retrieved_ids = [r["id"] for r in results]
        retrieved_texts = [r["text"] for r in results]
        
        # Compute deterministic metrics
        jf_metrics = calculate_judge_free_metrics(retrieved_ids, relevant_ids)
        for k in judge_free_totals:
            judge_free_totals[k] += jf_metrics[k]
            
        # Append for RAGAS dataset (v0.4.x nomenclature)
        ragas_data["user_input"].append(query)
        ragas_data["retrieved_contexts"].append(retrieved_texts)
        ragas_data["reference"].append(q_data["reference"])
        
        # Rate limiting delay for Qdrant/Embedding
        time.sleep(0.1)

    print("\n--- Judge-Free Metrics ---")
    n = len(eval_queries)
    avg_jf = {k: round(v / n, 4) for k, v in judge_free_totals.items()}
    print(avg_jf)

    print("\nInitializing RAGAS with Groq (llama3-8b-8192)...")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY missing from .env file!")
        
    groq_llm = ChatGroq(model_name="llama3-8b-8192")
    ragas_llm = LangchainLLMWrapper(groq_llm)
    
    dataset = Dataset.from_dict(ragas_data)
    
    print("Running RAGAS LLM-as-a-judge evaluation (this may take a few minutes)...")
    try:
        # RAGAS evaluate natively handles some retries, but we keep the batch size small
        ragas_results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            llm=ragas_llm
        )
        
        final_scores = ragas_results.copy()
        final_scores.update({"judge_free": avg_jf})
        
        print("\n--- Final Phase 1 Baseline Scores ---")
        print(ragas_results)
        
        with open(out_file, "w") as f:
            # ragas_results might be a custom dict, convert to standard dict
            json.dump(dict(final_scores), f, indent=2)
            
        print(f"\nBaseline logged successfully to {out_file}")
        
    except Exception as e:
        print(f"RAGAS Evaluation failed (likely a rate limit or parsing error). Error: {e}")
        print("Try running with a smaller limit: python eval/run_ragas.py --limit 5")

if __name__ == "__main__":
    main()