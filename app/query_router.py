from app.hybrid_retrieval import hybrid_search, DEFAULT_CANDIDATE_POOL

CONFIDENCE_THRESHOLD = 0.0
WIDENED_CANDIDATE_POOL = 60


def _top1_confidence(retrieved: list[dict]) -> float:
    return retrieved[0]["score"] if retrieved else float("-inf")


def route_query(collection: str, query_vector: list[float], query_text: str, top_k: int, candidate_pool: int = DEFAULT_CANDIDATE_POOL) -> dict:
    result = hybrid_search(collection, query_vector, query_text, top_k, candidate_pool)
    initial_confidence = _top1_confidence(result["retrieved"])
    triggered = initial_confidence < CONFIDENCE_THRESHOLD

    if triggered:
        result = hybrid_search(collection, query_vector, query_text, top_k, candidate_pool = WIDENED_CANDIDATE_POOL)

    result["routing"] = {
        "triggered": triggered,
        "initial_confidence": initial_confidence,
        "final_confidence": _top1_confidence(result["retrieved"]) if triggered else initial_confidence,
        "strategy": "widen_candidate_pool" if triggered else None,
    }
    return result
