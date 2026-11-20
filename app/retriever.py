import os
import re
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types

from app.config import settings
from app.pdf_parser import DocumentChunk, parse_pdf_manual

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 keyword retrieval, keeping alphanumeric words and numbers."""
    return re.findall(r"\b[\w\.\-%]+\b", text.lower())

class HybridRetriever:
    """
    Hybrid retriever combining:
    1. Dense Vector Search (ChromaDB + Gemini Embedding)
    2. Sparse Keyword Search (BM25Okapi)
    3. Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, pdf_path: Optional[str] = None, persist_dir: Optional[str] = None):
        self.pdf_path = pdf_path or settings.PDF_PATH
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embedding_model = settings.EMBEDDING_MODEL
        self.rrf_k = settings.RRF_K
        
        # Load PDF documents
        self.chunks: List[DocumentChunk] = parse_pdf_manual(self.pdf_path)
        self.chunk_by_id: Dict[str, DocumentChunk] = {c.chunk_id: c for c in self.chunks}
        self.chunk_by_page: Dict[int, DocumentChunk] = {c.page_number: c for c in self.chunks}

        # Setup Gemini GenAI Client
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

        # Setup BM25 Sparse Index
        self.tokenized_corpus = [tokenize_for_bm25(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Setup ChromaDB Dense Index
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._init_chroma_collection()

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using Google Gemini Embedding model."""
        if not self.gemini_client:
            raise RuntimeError("Gemini API key is required for embedding generation.")
        response = self.gemini_client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        return response.embeddings[0].values

    def _get_batch_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Batch generate embeddings with Gemini API."""
        if not self.gemini_client:
            raise RuntimeError("Gemini API key is required for embedding generation.")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=batch
            )
            for emb in response.embeddings:
                all_embeddings.append(emb.values)
        return all_embeddings

    def _init_chroma_collection(self):
        """Initializes or loads the ChromaDB collection."""
        collection_name = "bistro_manual_v1"
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            if collection.count() == len(self.chunks):
                return collection
            # Recreate if count mismatch
            self.chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Le Grand Bistro Culinary Operations 50-Page Manual"}
        )

        texts = [c.text for c in self.chunks]
        ids = [c.chunk_id for c in self.chunks]
        metadatas = [
            {
                "page_number": c.page_number,
                "chapter": c.chapter,
                "module": c.module,
                "has_recipe": c.metadata.get("has_recipe", False),
                "has_allergen": c.metadata.get("has_allergen", False),
                "has_haccp": c.metadata.get("has_haccp", False),
            }
            for c in self.chunks
        ]

        if self.gemini_client:
            embeddings = self._get_batch_embeddings(texts, batch_size=10)
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

        return collection

    def dense_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Dense semantic search using vector embeddings."""
        if not self.gemini_client:
            # Fallback for offline testing
            return []

        query_embedding = self._get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"]
        )

        ranked = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for rank_idx, chunk_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][rank_idx] if "distances" in results else 0.0
                ranked.append({
                    "chunk_id": chunk_id,
                    "dense_rank": rank_idx + 1,
                    "distance": dist,
                    "dense_score": 1.0 / (1.0 + dist)
                })
        return ranked

    def sparse_search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Sparse lexical search using BM25Okapi."""
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_pairs = list(enumerate(scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        ranked = []
        for rank_idx, (doc_idx, score) in enumerate(scored_pairs[:top_k]):
            chunk = self.chunks[doc_idx]
            ranked.append({
                "chunk_id": chunk.chunk_id,
                "sparse_rank": rank_idx + 1,
                "sparse_score": float(score)
            })
        return ranked

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Dense vector search
        2. Sparse BM25 keyword search
        3. Reciprocal Rank Fusion (RRF)
        """
        dense_results = self.dense_search(query, top_k=len(self.chunks))
        sparse_results = self.sparse_search(query, top_k=len(self.chunks))

        dense_map = {item["chunk_id"]: item for item in dense_results}
        sparse_map = {item["chunk_id"]: item for item in sparse_results}

        all_chunk_ids = set(self.chunk_by_id.keys())
        rrf_scores: Dict[str, float] = {}

        # Default rank if item not ranked in top list
        default_rank = len(self.chunks) + 1

        for cid in all_chunk_ids:
            d_rank = dense_map[cid]["dense_rank"] if cid in dense_map else default_rank
            s_rank = sparse_map[cid]["sparse_rank"] if cid in sparse_map else default_rank
            
            # Reciprocal Rank Fusion formula
            score = (1.0 / (self.rrf_k + d_rank)) + (1.0 / (self.rrf_k + s_rank))
            rrf_scores[cid] = score

        # Sort chunks by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for rank_idx, cid in enumerate(sorted_ids[:top_k]):
            chunk = self.chunk_by_id[cid]
            d_item = dense_map.get(cid, {})
            s_item = sparse_map.get(cid, {})

            results.append({
                "chunk_id": cid,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "chapter": chunk.chapter,
                "module": chunk.module,
                "rrf_rank": rank_idx + 1,
                "rrf_score": rrf_scores[cid],
                "dense_rank": d_item.get("dense_rank", default_rank),
                "dense_score": d_item.get("dense_score", 0.0),
                "sparse_rank": s_item.get("sparse_rank", default_rank),
                "sparse_score": s_item.get("sparse_score", 0.0),
                "metadata": chunk.metadata
            })

        return results
