import asyncio
import hashlib
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("RAPTOR.TreeBuilder")

class RAPTORNode(BaseModel):
    """Represents a node (leaf or cluster summary) in the RAPTOR tree hierarchy."""
    node_id: str = Field(..., description="Unique node identifier")
    level: int = Field(..., description="Tree level depth (0 = atomic leaf)")
    content: str = Field(..., description="Text content or abstractive summary")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding representation")
    children_ids: List[str] = Field(default_factory=list, description="IDs of child nodes summarized by this node")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

class LocalEmbeddingModel:
    """Local-first SentenceTransformers (BGE / E5) embedding manager."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def embed_text(self, text: str) -> List[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ('{self.model_name}'): {e}. Using deterministic local embedding.")
                self._model = False

        if self._model and not isinstance(self._model, bool):
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()

        # Fallback deterministic pseudo-embedding vector (dim=64)
        vec = [0.0] * 64
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 64
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 4) for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

class GMMClusterer:
    """Gaussian Mixture Model (GMM) Soft Clustering for RAPTOR tree nodes."""

    @staticmethod
    def cluster_nodes(nodes: List[RAPTORNode], target_cluster_size: int = 3) -> List[List[RAPTORNode]]:
        if len(nodes) <= target_cluster_size:
            return [nodes]

        num_clusters = max(2, math.ceil(len(nodes) / target_cluster_size))
        embeddings = [n.embedding for n in nodes if n.embedding]

        if len(embeddings) == len(nodes):
            try:
                import numpy as np
                from sklearn.mixture import GaussianMixture

                X = np.array(embeddings)
                n_components = min(num_clusters, len(nodes))
                gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
                gmm.fit(X)
                labels = gmm.predict(X)

                clusters: Dict[int, List[RAPTORNode]] = {}
                for label, node in zip(labels, nodes):
                    clusters.setdefault(int(label), []).append(node)
                return [c for c in clusters.values() if c]
            except Exception as err:
                logger.debug(f"GMM clustering fallback ({err}). Using distance-based grouping.")

        # Fallback grouping
        clusters_list: List[List[RAPTORNode]] = []
        for i in range(0, len(nodes), target_cluster_size):
            clusters_list.append(nodes[i:i + target_size if 'target_size' in locals() else i + target_cluster_size])
        return clusters_list

class RAPTORBuilder:
    """Production RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Tree Builder.
    Recursively builds multi-layer hierarchical summary trees and indexes them into vector DBs.
    """

    def __init__(
        self,
        embedding_model: Optional[Union[LocalEmbeddingModel, Callable[[str], List[float]]]] = None,
        summarizer_func: Optional[Callable[[List[str]], str]] = None,
        max_levels: int = 3,
        cluster_size: int = 3,
        qdrant_client: Any = None
    ):
        if isinstance(embedding_model, LocalEmbeddingModel):
            self.embed_func = embedding_model.embed_text
        elif callable(embedding_model):
            self.embed_func = embedding_model
        else:
            self._default_embedder = LocalEmbeddingModel()
            self.embed_func = self._default_embedder.embed_text

        self.summarizer_func = summarizer_func or self._default_summarizer
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.qdrant_client = qdrant_client

    def build_tree_index(self, document_id: str, leaf_chunks: List[Any]) -> List[RAPTORNode]:
        """Synchronously construct hierarchical RAPTOR tree index from leaf chunks."""
        if not leaf_chunks:
            return []

        all_nodes: List[RAPTORNode] = []
        current_level_nodes: List[RAPTORNode] = []

        # Level 0: Leaf Nodes
        for chunk in leaf_chunks:
            content = getattr(chunk, "content", str(chunk))
            chunk_id = getattr(chunk, "chunk_id", f"{document_id}_leaf_{len(all_nodes)}")
            emb = getattr(chunk, "embedding", None) or self.embed_func(content)
            
            meta = {}
            if hasattr(chunk, "metadata"):
                meta = chunk.metadata.model_dump() if hasattr(chunk.metadata, "model_dump") else getattr(chunk.metadata, "__dict__", {})

            node = RAPTORNode(
                node_id=chunk_id,
                level=0,
                content=content,
                embedding=emb,
                children_ids=[],
                metadata=meta
            )
            current_level_nodes.append(node)
            all_nodes.append(node)

        # Recursive Tree Construction (Levels 1 .. max_levels)
        current_level = 1
        while current_level < self.max_levels and len(current_level_nodes) > 1:
            clusters = GMMClusterer.cluster_nodes(current_level_nodes, target_cluster_size=self.cluster_size)
            next_level_nodes: List[RAPTORNode] = []

            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                children_ids = [n.node_id for n in cluster]
                cluster_contents = [n.content for n in cluster]
                summary_text = self.summarizer_func(cluster_contents)
                summary_emb = self.embed_func(summary_text)

                parent_node = RAPTORNode(
                    node_id=f"{document_id}_raptor_l{current_level}_c{idx}",
                    level=current_level,
                    content=f"[RAPTOR Level {current_level} Summary] {summary_text}",
                    embedding=summary_emb,
                    children_ids=children_ids,
                    metadata={"document_id": document_id, "level": current_level, "is_summary": True}
                )
                next_level_nodes.append(parent_node)
                all_nodes.append(parent_node)

            current_level_nodes = next_level_nodes
            current_level += 1

        if self.qdrant_client:
            self._upsert_to_qdrant(all_nodes)

        logger.info(f"RAPTOR Tree Built: {len(all_nodes)} nodes across {current_level} levels.")
        return all_nodes

    async def build_tree_index_async(self, document_id: str, leaf_chunks: List[Any]) -> List[RAPTORNode]:
        """Asynchronously construct hierarchical RAPTOR tree index."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.build_tree_index, document_id, leaf_chunks)

    def _upsert_to_qdrant(self, nodes: List[RAPTORNode]) -> None:
        """Upsert generated RAPTOR tree nodes into Qdrant Vector Store."""
        try:
            points = []
            for n in nodes:
                points.append({
                    "id": n.node_id,
                    "vector": n.embedding,
                    "payload": {"content": n.content, "level": n.level, **n.metadata}
                })
            if hasattr(self.qdrant_client, "upsert_points"):
                self.qdrant_client.upsert_points(points)
            elif hasattr(self.qdrant_client, "upsert"):
                self.qdrant_client.upsert(points)
            logger.info(f"Upserted {len(points)} nodes into Qdrant vector store.")
        except Exception as e:
            logger.warning(f"Qdrant vector store upsert warning: {e}")

    def _default_summarizer(self, text_blocks: List[str]) -> str:
        """Abstractive summarization fallback."""
        joined = " ".join(text_blocks)
        sentences = [s.strip() for s in joined.split(".") if len(s.strip()) > 10]
        selected = sentences[:3]
        return ". ".join(selected) + "." if selected else joined[:300]
