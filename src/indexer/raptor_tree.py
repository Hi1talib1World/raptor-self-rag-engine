import math
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel
    class RAPTORNode(BaseModel):
        node_id: str
        level: int
        content: str
        embedding: Optional[List[float]] = None
        children_ids: List[str] = []
        metadata: Dict[str, Any] = {}
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class RAPTORNode:
        node_id: str
        level: int
        content: str
        embedding: Optional[List[float]] = None
        children_ids: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

class RAPTORBuilder:
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."""

    def __init__(self, embed_func=None, max_levels: int = 3, cluster_size: int = 3):
        self.embed_func = embed_func or self._mock_embed
        self.max_levels = max_levels
        self.cluster_size = cluster_size

    def build_tree_index(self, document_id: str, leaf_chunks: List[Any]) -> List[RAPTORNode]:
        if not leaf_chunks:
            return []

        all_nodes: List[RAPTORNode] = []
        current_level_nodes: List[RAPTORNode] = []

        # Level 0: Leaf Nodes
        for chunk in leaf_chunks:
            emb = getattr(chunk, "embedding", None) or self.embed_func(chunk.content)
            meta = chunk.metadata.model_dump() if hasattr(chunk.metadata, "model_dump") else chunk.metadata.__dict__
            node = RAPTORNode(
                node_id=chunk.chunk_id,
                level=0,
                content=chunk.content,
                embedding=emb,
                children_ids=[],
                metadata=meta
            )
            current_level_nodes.append(node)
            all_nodes.append(node)

        # Recursive Tree Construction (Levels 1 .. max_levels)
        current_level = 1
        while current_level < self.max_levels and len(current_level_nodes) > 1:
            clusters = self._cluster_nodes(current_level_nodes, self.cluster_size)
            next_level_nodes: List[RAPTORNode] = []

            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                children_ids = [n.node_id for n in cluster]
                summary_text = f"[Level {current_level} Summary] " + " ".join([n.content[:100] for n in cluster])
                summary_emb = self.embed_func(summary_text)

                parent_node = RAPTORNode(
                    node_id=f"{document_id}_raptor_l{current_level}_c{idx}",
                    level=current_level,
                    content=summary_text,
                    embedding=summary_emb,
                    children_ids=children_ids,
                    metadata={"document_id": document_id, "level": current_level, "is_summary": True}
                )
                next_level_nodes.append(parent_node)
                all_nodes.append(parent_node)

            current_level_nodes = next_level_nodes
            current_level += 1

        return all_nodes

    def _cluster_nodes(self, nodes: List[RAPTORNode], target_size: int) -> List[List[RAPTORNode]]:
        num_clusters = math.ceil(len(nodes) / target_size)
        return [nodes[i * target_size: min(len(nodes), (i + 1) * target_size)] for i in range(num_clusters)]

    def _mock_embed(self, text: str) -> List[float]:
        import hashlib
        vec = [0.0] * 64
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
