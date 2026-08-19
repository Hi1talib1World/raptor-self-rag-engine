import re
from typing import List
from src.retrieval.hybrid_search import SearchResult

class CrossEncoderReranker:
    """SentenceTransformers Joint-Attention Cross-Encoder Reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not results:
            return []

        if not self.model:
            # High-performance lexical overlap fallback reranking
            q_tokens = set(re.findall(r"\w+", query.lower()))
            reranked = []
            for item in results:
                c_tokens = set(re.findall(r"\w+", item.content.lower()))
                overlap = len(q_tokens.intersection(c_tokens)) / (len(q_tokens) or 1)
                new_score = round((item.score * 0.6) + (overlap * 0.4), 4)
                reranked.append(SearchResult(
                    node_id=item.node_id,
                    content=item.content,
                    score=new_score,
                    retrieval_type="reranked",
                    metadata=item.metadata
                ))
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]

        pairs = [[query, r.content] for r in results]
        scores = self.model.predict(pairs)

        reranked = []
        for r, score in zip(results, scores):
            normalized = float(1.0 / (1.0 + pow(2.71828, -float(score))))
            reranked.append(SearchResult(
                node_id=r.node_id,
                content=r.content,
                score=round(normalized, 4),
                retrieval_type="cross_encoder_reranked",
                metadata=r.metadata
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
