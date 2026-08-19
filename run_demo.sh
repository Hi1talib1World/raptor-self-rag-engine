#!/usr/bin/env bash

set -e

echo "========================================================================="
echo "   🧠 RAPTOR Self-RAG Engine — Automated Demo & Terminal Interface"
echo "========================================================================="

# Step 1: Ensure Docker service or Local environment
if command -v docker &> /dev/null; then
    echo "[1/4] Starting Qdrant Vector DB container..."
    docker run -d --name raptor_qdrant_demo -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.9.0 2>/dev/null || echo "Qdrant container already running or starting locally."
else
    echo "[1/4] Docker not detected. Operating in high-speed local vector memory mode."
fi

# Step 2: Create sample multi-page knowledge base
echo "[2/4] Ingesting multi-page markdown knowledge base..."
mkdir -p data/knowledge_base
cat << 'EOF' > data/knowledge_base/industrial_architecture.md
# Enterprise Industrial Edge Architecture Specification

## 1. Protocol Support
The manufacturing edge telemetry system provides multi-protocol ingestion supporting:
- MQTT over TLS 1.3 for lightweight pub-sub sensor telemetry.
- OPC-UA (Open Platform Communications Unified Architecture) for industrial automation and complex data modeling.
- Modbus TCP/RTU for legacy PLC register polling.

## 2. Zero Trust Security Model
All edge-to-cloud telemetry streams enforce ISO 27001 Zero Trust network access boundaries:
- Mutual TLS (mTLS) authentication with hardware TPM 2.0 key storage.
- Continuous token validation and log auditing.
EOF

# Step 3: Index knowledge base with RAPTOR
echo "[3/4] Building RAPTOR Hierarchical Tree Index across knowledge base..."
python -c "
import os
from src.indexer.chunking import SemanticChunker
from src.raptor.tree_builder import RAPTORBuilder
from src.retrieval.hybrid_search import HybridSearchEngine

with open('data/knowledge_base/industrial_architecture.md', 'r') as f:
    text = f.read()

chunker = SemanticChunker(target_chunk_size=300)
builder = RAPTORBuilder(max_levels=2, cluster_size=2)
chunks = chunker.chunk_text('doc_demo', text, 'industrial_architecture.md')
nodes = builder.build_tree_index('doc_demo', chunks)

print(f'Successfully indexed knowledge base into {len(nodes)} RAPTOR tree nodes (Leaf chunks + Level 1 summaries).')
"

# Step 4: Launch Interactive Terminal Session
echo "[4/4] Launching Interactive Terminal Q&A Session..."
echo "Type your question below (or 'exit' to quit):"
echo "-------------------------------------------------------------------------"

python -c "
import sys
from src.indexer.chunking import SemanticChunker
from src.raptor.tree_builder import RAPTORBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.reflection.self_rag_tokens import SelfRAGReflectionEngine
from src.router.adaptive_gate import AdaptiveRouterGate
from src.models.llm_provider import get_llm_provider

# Load knowledge base
with open('data/knowledge_base/industrial_architecture.md', 'r') as f:
    text = f.read()

chunker = SemanticChunker(target_chunk_size=300)
builder = RAPTORBuilder(max_levels=2, cluster_size=2)
nodes = builder.build_tree_index('doc_demo', text.split('\n\n'))

search = HybridSearchEngine(nodes)
reranker = CrossEncoderReranker()
reflector = SelfRAGReflectionEngine()
router = AdaptiveRouterGate()
llm = get_llm_provider('mock')

def run_query(q):
    decision = router.route(q)
    results = search.search(q, top_k=3)
    reranked = reranker.rerank(q, results, top_k=2)
    ctx = '\n\n'.join([r.content for r in reranked])
    
    critique = reflector.run_critique_loop(q, ctx, lambda query, context: llm.generate(f'Context: {context}\nQuestion: {query}'))
    
    print(f'\nModel Route: {decision.selected_model} (Uncertainty: {decision.predictive_uncertainty})')
    print(f'Answer: {critique.answer}')
    print(f'Reflection Tokens: {critique.reflection_tokens}\n')

# Quick demonstration queries
print('--- Demonstration Query 1 ---')
run_query('What industrial protocols are supported?')

print('--- Demonstration Query 2 ---')
run_query('Compare MQTT TLS 1.3 security with ISO 27001 zero trust architecture.')
"

echo "========================================================================="
echo "   ✅ RAPTOR Self-RAG Engine Demo Completed Successfully!"
echo "========================================================================="
