import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None
    collection: str = "docs"

class RAGUpsertresult(pydantic.BaseModel):
    ingested: int

class RetrievedChunk(pydantic.BaseModel):
    text: str
    source: str
    score: float

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    retrieved: list[RetrievedChunk] = []

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
