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
