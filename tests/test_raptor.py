import unittest
from typing import List, Dict, Any
from src.raptor.tree_builder import RAPTORBuilder, RAPTORNode, LocalEmbeddingModel, GMMClusterer
from src.indexer.chunking import SemanticChunker

class MockVectorStore:
    """Mock Vector Store representing Qdrant storage."""
    def __init__(self):
        self.indexed_nodes: Dict[str, Any] = {}

    def upsert_points(self, points: List[Dict[str, Any]]):
        for p in points:
            self.indexed_nodes[p["id"]] = p

class TestRAPTORTreeBuilder(unittest.TestCase):
    def setUp(self):
        self.embedder = LocalEmbeddingModel()
        self.mock_qdrant = MockVectorStore()
        self.builder = RAPTORBuilder(
            embedding_model=self.embedder,
            max_levels=3,
            cluster_size=2,
            qdrant_client=self.mock_qdrant
        )

    def test_leaf_chunk_ingestion(self):
        chunker = SemanticChunker(target_chunk_size=150)
        content = "MQTT is an ISO standard pub-sub messaging protocol.\n\nOPC-UA is an industrial automation protocol."
        chunks = chunker.chunk_text("doc_test", content, "protocols.md")
        self.assertGreaterEqual(len(chunks), 1)

    def test_gmm_clusterer(self):
        n1 = RAPTORNode(node_id="n1", level=0, content="MQTT protocol details", embedding=[0.1, 0.2, 0.3])
        n2 = RAPTORNode(node_id="n2", level=0, content="OPC UA automation protocol", embedding=[0.15, 0.25, 0.35])
        n3 = RAPTORNode(node_id="n3", level=0, content="Database SQL queries", embedding=[0.9, 0.8, 0.7])

        clusters = GMMClusterer.cluster_nodes([n1, n2, n3], target_cluster_size=2)
        self.assertGreaterEqual(len(clusters), 1)

    def test_hierarchical_tree_indexing(self):
        leaves = [
            RAPTORNode(node_id="leaf_1", level=0, content="MQTT telemetry data transport over TCP/IP.", embedding=self.embedder.embed_text("MQTT telemetry transport")),
            RAPTORNode(node_id="leaf_2", level=0, content="TLS 1.3 encryption secures MQTT payload data.", embedding=self.embedder.embed_text("TLS encryption payload")),
            RAPTORNode(node_id="leaf_3", level=0, content="OPC-UA provides object-oriented industrial data modeling.", embedding=self.embedder.embed_text("OPC-UA data modeling")),
            RAPTORNode(node_id="leaf_4", level=0, content="Modbus protocol operates over serial and TCP networks.", embedding=self.embedder.embed_text("Modbus serial TCP"))
        ]

        tree_nodes = self.builder.build_tree_index("doc_raptor_test", leaves)
        
        # Verify leaves (level 0) and higher level summary nodes exist
        self.assertGreater(len(tree_nodes), len(leaves))
        self.assertTrue(any(n.level == 0 for n in tree_nodes))
        self.assertTrue(any(n.level == 1 for n in tree_nodes))

        # Check mock vector store integration
        self.assertGreater(len(self.mock_qdrant.indexed_nodes), 0)
        self.assertIn("doc_raptor_test_leaf_0", self.mock_qdrant.indexed_nodes)

if __name__ == "__main__":
    unittest.main()
