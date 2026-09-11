import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai

from dotenv import load_dotenv
import uuid  # to generate unique IDs
import os
import datetime

from app.data_loader import load_and_chunk_pdf, embed_texts
from app.vector_db import QdrantStorage
from app.custom_types import RAGChunkAndSrc, RAGUpsertresult, RAGSearchResult
from app.hybrid_retrieval import hybrid_search, invalidate_bm25_cache
from app.query_router import route_query

load_dotenv() # load environment variables from .env file

inngest_client = inngest.Inngest(
    app_id = "rag_app",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer() # defines the types of different variable 
)

@inngest_client.create_function(
    fn_id = "RAG: Ingest PDF",
    trigger = inngest.TriggerEvent(event = "rag/ingest_pdf"),
    throttle = inngest.Throttle(limit = 2, period = datetime.timedelta(minutes = 1)),
    rate_limit = inngest.RateLimit(limit = 2, period = datetime.timedelta(hours = 4), key = "event.data.source_id")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        collection = ctx.event.data.get("collection", "docs")
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks = chunks, source_id = source_id, collection = collection)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertresult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage(collection = chunks_and_src.collection).upsert(ids, vecs, payloads)
        invalidate_bm25_cache(chunks_and_src.collection)
        return RAGUpsertresult(ingested = len(chunks))
    
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type = RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type = RAGUpsertresult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id = "RAG: Query PDF",
    trigger = inngest.TriggerEvent(event = "rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int, collection: str, retrieval_mode: str, enable_routing: bool) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        if retrieval_mode == "dense":
            found = QdrantStorage(collection = collection).search(query_vec, top_k)
            found["routing"] = {"triggered": False, "strategy": None}
        elif enable_routing:
            found = route_query(collection, query_vec, question, top_k)
        else:
            found = hybrid_search(collection, query_vec, question, top_k)
            found["routing"] = {"triggered": False, "strategy": None}
        return RAGSearchResult(contexts = found["contexts"], scores = found["scores"], sources = found["sources"], retrieved = found["retrieved"], routing = found["routing"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    collection = ctx.event.data.get("collection", "docs")
    retrieval_mode = ctx.event.data.get("retrieval_mode", "hybrid")

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k, collection, retrieval_mode), output_type = RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_context = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key = os.getenv("OPENAI_API_KEY"),
        model = "gpt-4o-mini",
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter = adapter,
        body = {
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_context}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
        "retrieved_chunks": [c.model_dump() for c in found.retrieved],
    }


    
app = FastAPI()


inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf, rag_query_pdf_ai])
