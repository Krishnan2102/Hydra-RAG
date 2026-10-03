from typing import Any, Dict, List


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float] = None,
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF).
    score(d) = sum_m [ weight_m / (k + rank_m(d)) ]
    """
    if weights is None:
        weights = {name: 1.0 for name in ranked_lists}

    rrf_scores = {}
    doc_payloads = {}
    rank_details = {}

    for leg_name, items in ranked_lists.items():
        weight = weights.get(leg_name, 1.0)
        for rank, item in enumerate(items, start=1):
            doc_id = item["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_payloads[doc_id] = item
                rank_details[doc_id] = {}

            rrf_scores[doc_id] += weight * (1.0 / (k + rank))
            rank_details[doc_id][f"{leg_name}_rank"] = rank
            rank_details[doc_id][f"{leg_name}_score"] = item.get("score")

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_docs:
        entry = dict(doc_payloads[doc_id])
        entry["fused_score"] = round(score, 6)
        entry["ranks"] = rank_details[doc_id]
        results.append(entry)

    return results


def weighted_linear_fusion(
    ranked_lists: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Weighted Linear Combination with Min-Max score normalization.
    """
    if weights is None:
        weights = {name: 1.0 for name in ranked_lists}

    total_w = sum(weights.values()) or 1.0
    norm_weights = {k: v / total_w for k, v in weights.items()}

    normalized_scores = {}
    doc_payloads = {}

    for leg_name, items in ranked_lists.items():
        if not items:
            continue
        scores = [item["score"] for item in items]
        min_s = min(scores)
        max_s = max(scores)
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0

        for item in items:
            doc_id = item["id"]
            if doc_id not in normalized_scores:
                normalized_scores[doc_id] = 0.0
                doc_payloads[doc_id] = item

            norm_val = (item["score"] - min_s) / range_s
            normalized_scores[doc_id] += norm_weights[leg_name] * norm_val

    sorted_docs = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_docs:
        entry = dict(doc_payloads[doc_id])
        entry["fused_score"] = round(score, 6)
        results.append(entry)

    return results