import math
import re
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
    class SearchResult(BaseModel):
        node_id: str
        content: str
        score: float
        retrieval_type: str = "hybrid"
        metadata: Dict[str, Any] = {}
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class SearchResult:
        node_id: str
        content: str
        score: float
        retrieval_type: str = "hybrid"
        metadata: Dict[str, Any] = field(default_factory=dict)

class HybridSearchEngine:
    """BM25 Lexical + Dense Vector Hybrid Search Engine with RRF Fusion."""

    def __init__(self, nodes: List[Any] = None, embed_func=None):
        self.nodes = nodes or []
        self.embed_func = embed_func or self._mock_embed

    def index_nodes(self, nodes: List[Any]):
        self.nodes = nodes

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.nodes:
            return []

        # 1. Dense Vector Search
        q_vec = self.embed_func(query)
        dense_scores = []
        for n in self.nodes:
            n_vec = getattr(n, "embedding", None) or self.embed_func(n.content)
            sim = self._cosine_sim(q_vec, n_vec)
            meta = n.metadata.model_dump() if hasattr(n.metadata, "model_dump") else n.metadata
            dense_scores.append(SearchResult(node_id=n.node_id, content=n.content, score=sim, retrieval_type="dense", metadata=meta))

        dense_scores.sort(key=lambda x: x.score, reverse=True)

        # 2. Sparse Lexical Search (BM25 term overlap)
        q_tokens = set(re.findall(r"\w+", query.lower()))
        sparse_scores = []
        for n in self.nodes:
            n_tokens = set(re.findall(r"\w+", n.content.lower()))
            overlap = len(q_tokens.intersection(n_tokens)) / (len(q_tokens) or 1)
            meta = n.metadata.model_dump() if hasattr(n.metadata, "model_dump") else n.metadata
            sparse_scores.append(SearchResult(node_id=n.node_id, content=n.content, score=overlap, retrieval_type="sparse", metadata=meta))

        sparse_scores.sort(key=lambda x: x.score, reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        return self._rrf_fusion([dense_scores[:top_k * 2], sparse_scores[:top_k * 2]], top_k=top_k)

    def _rrf_fusion(self, result_lists: List[List[SearchResult]], top_k: int, rrf_k: int = 60) -> List[SearchResult]:
        rrf_map: Dict[str, float] = {}
        node_map: Dict[str, SearchResult] = {}

        for res_list in result_lists:
            for rank, item in enumerate(res_list):
                nid = item.node_id
                node_map[nid] = item
                rrf_map[nid] = rrf_map.get(nid, 0.0) + (1.0 / (rrf_k + (rank + 1)))

        max_s = max(rrf_map.values()) if rrf_map else 1.0
        fused = []
        for nid, score in rrf_map.items():
            item = node_map[nid]
            fused.append(SearchResult(
                node_id=item.node_id,
                content=item.content,
                score=round(score / max_s, 4),
                retrieval_type="rrf_hybrid",
                metadata=item.metadata
            ))

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def _cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        min_l = min(len(v1), len(v2))
        dot = sum(a * b for a, b in zip(v1[:min_l], v2[:min_l]))
        n1 = math.sqrt(sum(a * a for a in v1[:min_l])) or 1e-9
        n2 = math.sqrt(sum(b * b for b in v2[:min_l])) or 1e-9
        return float(dot / (n1 * n2))

    def _mock_embed(self, text: str) -> List[float]:
        import hashlib
        vec = [0.0] * 64
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
