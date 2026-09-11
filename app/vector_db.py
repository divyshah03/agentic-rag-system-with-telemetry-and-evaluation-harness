import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url = None, api_key = None, collection = "docs", dim = 3072):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url = url, api_key = api_key, timeout = 30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config = VectorParams(size = dim, distance = Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points = points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name = self.collection,
            query = query_vector,
            with_payload = True,
            limit = top_k
        )

        contexts = []
        sources = set()
        scores = []
        retrieved = []

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                  contexts.append(text)
                  sources.add(source)
                  scores.append(r.score)
                  retrieved.append({"id": r.id, "text": text, "source": source, "score": r.score})
        return {"contexts": contexts,"sources": list(sources), "scores": scores, "retrieved": retrieved}

    def scroll_all(self):
        documents = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name = self.collection,
                with_payload = True,
                with_vectors = False,
                limit = 256,
                offset = offset
            )
            for p in points:
                payload = getattr(p, "payload", None) or {}
                text = payload.get("text", "")
                if text:
                    documents.append({"id": p.id, "text": text, "source": payload.get("source", "")})
            if offset is None:
                break
        return documents

          



        
   