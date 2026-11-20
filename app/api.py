from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time

from app.config import settings, EXACT_ABSTENTION_MESSAGE
from app.rag_engine import get_rag_engine

app = FastAPI(
    title="Grand Bistro Strict Closed-Domain RAG API",
    description="Factual RAG system restricted exclusively to the Grand Bistro Culinary Recipe Manual PDF.",
    version="1.0.0"
)

# Singleton engine instance
engine = get_rag_engine()
retriever = engine.retriever

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query regarding the culinary manual")
    top_k: int = Field(5, ge=1, le=20, description="Number of passages to retrieve for hybrid search")

class ChunkResponse(BaseModel):
    chunk_id: str
    page_number: int
    rrf_rank: int
    rrf_score: float
    dense_rank: int
    sparse_rank: int
    text_snippet: str
    module: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[int]
    is_abstained: bool
    flagged_injection: bool
    latency_seconds: float
    retrieved_chunks: List[ChunkResponse]

@app.get("/")
def root():
    return {
        "system": "Le Grand Bistro Closed-Domain RAG",
        "status": "online",
        "knowledge_source": "grand_bistro_culinary_recipe_manual_50p.pdf",
        "total_pages": len(retriever.chunks),
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pdf_loaded": len(retriever.chunks) == 50,
        "chroma_collection_count": retriever.collection.count(),
        "bm25_indexed_docs": len(retriever.tokenized_corpus)
    }

@app.get("/stats")
def get_stats():
    return {
        "pdf_path": settings.PDF_PATH,
        "total_pages": len(retriever.chunks),
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "rrf_k": settings.RRF_K,
        "special_pages": {
            "page_10": "Recipe Card 2.4: Signature Truffle Ganache",
            "page_25": "Section 5.1: Autumn Tasting Menu Allergen Matrix Table",
            "page_40": "Section 7.2: HACCP Food Safety & Sous-Vide Duck Breast Pasteurization"
        }
    }

@app.get("/page/{page_num}")
def get_page_text(page_num: int):
    """Retrieve raw text of a specific page for auditability."""
    if page_num < 1 or page_num > len(retriever.chunks):
        raise HTTPException(status_code=404, detail=f"Page {page_num} not found. Manual has 1-{len(retriever.chunks)} pages.")
    chunk = retriever.chunk_by_page[page_num]
    return {
        "page_number": page_num,
        "chapter": chunk.chapter,
        "module": chunk.module,
        "metadata": chunk.metadata,
        "text": chunk.text
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.time()
    result = engine.answer_question(request.query, top_k=request.top_k)
    elapsed = time.time() - start_time

    formatted_chunks = [
        ChunkResponse(
            chunk_id=c["chunk_id"],
            page_number=c["page_number"],
            rrf_rank=c["rrf_rank"],
            rrf_score=c["rrf_score"],
            dense_rank=c["dense_rank"],
            sparse_rank=c["sparse_rank"],
            text_snippet=c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            module=c.get("module")
        )
        for c in result["retrieved_chunks"]
    ]

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result["citations"],
        is_abstained=result["is_abstained"],
        flagged_injection=result["flagged_injection"],
        latency_seconds=round(elapsed, 3),
        retrieved_chunks=formatted_chunks
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_PORT)
