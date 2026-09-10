import re

from rank_bm25 import BM25Okapi

from app.vector_db import QdrantStorage

RRF_K = 60
DEFAULT_CANDIDATE_POOL = 20
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_bm25_cache: dict[str, tuple] = {}


def _get_bm25_index(collection: str):
    cached = _bm25_cache.get(collection)
    if cached is not None:
        return cached
    documents = QdrantStorage(collection = collection).scroll_all()
    tokenized_corpus = [_tokenize(doc["text"]) for doc in documents]
    index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _bm25_cache[collection] = (index, documents)
    return index, documents


def invalidate_bm25_cache(collection: str) -> None:
    _bm25_cache.pop(collection, None)


def _bm25_candidates(collection: str, query_text: str, pool_size: int) -> list[dict]:
    index, documents = _get_bm25_index(collection)
    if index is None or not documents:
        return []
    scores = index.get_scores(_tokenize(query_text))
    ranked = sorted(range(len(documents)), key = lambda i: scores[i], reverse = True)
    return [documents[i] for i in ranked[:pool_size] if scores[i] > 0]


def _reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    fused_scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start = 1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused_scores


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name = RERANKER_MODEL)
    return _reranker
