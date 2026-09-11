from app.hybrid_retrieval import hybrid_search, DEFAULT_CANDIDATE_POOL

CONFIDENCE_THRESHOLD = 0.0
WIDENED_CANDIDATE_POOL = 60


def _top1_confidence(retrieved: list[dict]) -> float:
    return retrieved[0]["score"] if retrieved else float("-inf")
