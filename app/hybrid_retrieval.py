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


def hybrid_search(collection: str, query_vector: list[float], query_text: str, top_k: int, candidate_pool: int = DEFAULT_CANDIDATE_POOL) -> dict:
    pool_size = max(candidate_pool, top_k)

    dense = QdrantStorage(collection = collection).search(query_vector, pool_size)["retrieved"]
    bm25 = _bm25_candidates(collection, query_text, pool_size)

    by_id = {doc["id"]: doc for doc in bm25}
    by_id.update({doc["id"]: doc for doc in dense})

    fused_scores = _reciprocal_rank_fusion([
        [doc["id"] for doc in dense],
        [doc["id"] for doc in bm25],
    ])
    fused_ids = sorted(fused_scores, key = lambda doc_id: fused_scores[doc_id], reverse = True)
    candidates = [by_id[doc_id] for doc_id in fused_ids[:pool_size]]

    if not candidates:
        return {"contexts": [], "sources": [], "scores": [], "retrieved": []}

    reranker = _get_reranker()
    rerank_scores = list(reranker.rerank(query_text, [c["text"] for c in candidates]))
    ranked = sorted(zip(candidates, rerank_scores), key = lambda pair: pair[1], reverse = True)[:top_k]

    return {
        "contexts": [c["text"] for c, _ in ranked],
        "sources": list({c["source"] for c, _ in ranked}),
        "scores": [float(score) for _, score in ranked],
        "retrieved": [{"text": c["text"], "source": c["source"], "score": float(score)} for c, score in ranked],
    }
