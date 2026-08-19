import unittest
from src.indexer.chunking import SemanticChunker
from src.indexer.raptor_tree import RAPTORBuilder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.reflection.self_rag_tokens import SelfRAGReflectionEngine
from src.gateway.adaptive_router import AdaptiveQueryRouter
from src.models.llm_provider import get_llm_provider, OllamaLLMProvider, vLLMProvider
from benchmarks.eval_rag import RAGMetricsEvaluator

from src.raptor.tree_builder import RAPTORBuilder as TreeBuilder
from src.router.adaptive_gate import AdaptiveRouterGate

class TestRAPTORSelfRAGEngine(unittest.TestCase):
    def test_chunker_and_raptor(self):
        chunker = SemanticChunker(target_chunk_size=200)
        chunks = chunker.chunk_text("doc_1", "MQTT is a protocol.\n\nOPC-UA is another protocol.")
        self.assertGreaterEqual(len(chunks), 1)

        builder = RAPTORBuilder(max_levels=2, cluster_size=2)
        nodes = builder.build_tree_index("doc_1", chunks)
        self.assertGreaterEqual(len(nodes), len(chunks))

    def test_new_raptor_and_gate_modules(self):
        import asyncio
        tb = TreeBuilder(max_levels=2, cluster_size=2)
        nodes = asyncio.run(tb.build_tree_index_async("doc_test", ["Passage A content.", "Passage B content."]))
        self.assertGreaterEqual(len(nodes), 2)

        gate = AdaptiveRouterGate()
        res = asyncio.run(gate.route_async("What is MQTT?"))
        self.assertFalse(res.is_frontier)
        self.assertEqual(res.selected_model, "ollama/llama3")

    def test_hybrid_search_and_reranker(self):
        chunker = SemanticChunker(target_chunk_size=200)
        chunks = chunker.chunk_text("doc_1", "MQTT protocol for IoT.\n\nDatabase SQL queries.")
        builder = RAPTORBuilder(max_levels=2, cluster_size=2)
        nodes = builder.build_tree_index("doc_1", chunks)

        engine = HybridSearchEngine(nodes)
        results = engine.search("MQTT protocol", top_k=2)
        self.assertGreaterEqual(len(results), 1)

        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("MQTT protocol", results)
        self.assertGreaterEqual(len(reranked), 1)

    def test_self_rag_critique_loop(self):
        reflector = SelfRAGReflectionEngine()
        res = reflector.run_critique_loop("What is MQTT?", "MQTT is an IoT protocol.", lambda q, c: "MQTT is an IoT protocol.")
        self.assertTrue(res.should_retrieve)
        self.assertTrue(res.is_supported)
        self.assertIn("[Retrieve: YES]", res.reflection_tokens)

    def test_adaptive_router(self):
        router = AdaptiveQueryRouter()
        decision = router.route_query("What is MQTT?")
        self.assertFalse(decision.is_frontier_model)
        self.assertEqual(decision.target_model, "meta-llama/Meta-Llama-3-8B-Instruct")

    def test_llm_providers(self):
        provider = get_llm_provider("mock")
        res = provider.generate("Test prompt")
        self.assertTrue(len(res) > 0)

        ollama_p = OllamaLLMProvider()
        res_ollama = ollama_p.generate("Test Ollama")
        self.assertIn("Ollama Fallback", res_ollama)

        vllm_p = vLLMProvider()
        res_vllm = vllm_p.generate("Test vLLM")
        self.assertIn("vLLM Fallback", res_vllm)

    def test_metrics_evaluator(self):
        m = RAGMetricsEvaluator.evaluate_pair(
            query="What is MQTT?",
            answer="MQTT is a protocol.",
            retrieved_contexts=["MQTT is a messaging protocol."],
            expected_answer="MQTT is a messaging protocol.",
            ground_truth_sources=["doc.md"]
        )
        self.assertGreater(m["overall_rag_score"], 0.0)
        self.assertIn("faithfulness", m)

if __name__ == "__main__":
    unittest.main()
