import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.indexer.chunking import SemanticChunker
from src.indexer.raptor_tree import RAPTORBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.reflection.self_rag_tokens import SelfRAGReflectionEngine
from src.gateway.adaptive_router import AdaptiveQueryRouter
from src.models.llm_provider import get_llm_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAPTOR.Server")

app = FastAPI(
    title="RAPTOR Self-RAG Engine",
    description="Enterprise RAG 2.0 system featuring RAPTOR hierarchical retrieval, Self-RAG reflection tokens, and adaptive query routing.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instances
chunker = SemanticChunker(target_chunk_size=400)
raptor_builder = RAPTORBuilder(max_levels=2, cluster_size=2)
search_engine = HybridSearchEngine()
reranker = CrossEncoderReranker()
reflection_engine = SelfRAGReflectionEngine()
router = AdaptiveQueryRouter()

class IngestRequest(BaseModel):
    document_id: str
    content: str
    source: str = "user_document.txt"

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"
    top_k: int = 5
    enable_raptor: bool = True
    enable_self_rag: bool = True

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "RAPTOR Self-RAG Engine", "version": "2.0.0"}

@app.post("/ingest")
def ingest_document(req: IngestRequest):
    try:
        leaf_chunks = chunker.chunk_text(req.document_id, req.content, req.source)
        raptor_nodes = raptor_builder.build_tree_index(req.document_id, leaf_chunks)
        search_engine.index_nodes(raptor_nodes)
        
        return {
            "status": "success",
            "document_id": req.document_id,
            "leaf_chunks_created": len(leaf_chunks),
            "raptor_tree_nodes": len(raptor_nodes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
def query_engine(req: QueryRequest):
    try:
        start_t = time.time()

        # 1. Adaptive Routing
        routing = router.route_query(req.query)

        # 2. Hybrid Retrieval + Cross-Encoder Rerank
        raw_results = search_engine.search(req.query, top_k=req.top_k * 2)
        reranked = reranker.rerank(req.query, raw_results, top_k=req.top_k)

        context_blocks = [f"[{r.metadata.get('source', 'doc')}] {r.content}" for r in reranked]
        assembled_context = "\n\n".join(context_blocks)

        # 3. Self-RAG Generation & Reflection Loop
        llm_provider = get_llm_provider(os.getenv("LLM_PROVIDER", "mock"))

        def base_generator(q, c):
            if not c.strip():
                return "Insufficient information in the retrieved context to answer this question."
            prompt = f"Context:\n{c}\n\nUser Question: {q}\nProvide a grounded answer based strictly on context."
            return llm_provider.generate(prompt)

        if req.enable_self_rag:
            self_rag_output = reflection_engine.run_critique_loop(req.query, assembled_context, base_generator)
            final_answer = self_rag_output.answer
            reflection_tokens = self_rag_output.reflection_tokens
        else:
            final_answer = base_generator(req.query, assembled_context)
            reflection_tokens = []

        latency = round((time.time() - start_t) * 1000.0, 2)

        return {
            "query": req.query,
            "answer": final_answer,
            "routing": routing.model_dump() if hasattr(routing, "model_dump") else routing.__dict__,
            "retrieved_nodes_count": len(reranked),
            "reflection_tokens": reflection_tokens,
            "latency_ms": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

def run_benchmark():
    """Run benchmark evaluation suite over benchmarks/eval_dataset.json."""
    dataset_path = "benchmarks/eval_dataset.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"=== Running RAPTOR Self-RAG Benchmark Suite ({len(data)} test pairs) ===")
        for item in data:
            print(f"Query [{item['id']}]: {item['query']}")
            print(f"  Expected Token: {item['expected_reflection_token']} | Difficulty: {item['difficulty']}\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
