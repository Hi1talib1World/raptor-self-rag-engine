import unittest
from src.router.adaptive_gate import AdaptiveRouterGate, RoutingDecision

class TestAdaptiveRouterGate(unittest.TestCase):
    def setUp(self):
        self.router = AdaptiveRouterGate(
            local_ollama_model="ollama/llama3",
            frontier_model="gpt-4o",
            uncertainty_threshold=0.35,
            daily_budget_usd=10.0
        )

    def test_simple_factual_query_routing(self):
        decision = self.router.route("What is MQTT?")
        self.assertFalse(decision.is_frontier)
        self.assertEqual(decision.selected_model, "ollama/llama3")
        self.assertEqual(decision.provider, "ollama")

    def test_complex_multihop_query_routing(self):
        query = "Compare and contrast the security implications of MQTT TLS 1.3 vs OPC-UA zero trust architecture."
        decision = self.router.route(query)
        self.assertTrue(decision.is_frontier)
        self.assertEqual(decision.selected_model, "gpt-4o")

    def test_logprob_entropy_uncertainty_calculation(self):
        # Low logprob values (high probability) -> Low uncertainty
        decision_low = self.router.route("Simple question", sample_logprobs=[-0.01, -0.02, -0.05])
        self.assertLess(decision_low.predictive_uncertainty, 0.35)
        self.assertFalse(decision_low.is_frontier)

        # High logprob values (low probability) -> High uncertainty
        decision_high = self.router.route("Ambiguous query", sample_logprobs=[-3.5, -4.2, -2.9])
        self.assertGreaterEqual(decision_high.predictive_uncertainty, 0.35)
        self.assertTrue(decision_high.is_frontier)

    def test_budget_exceeded_fallback(self):
        # Force budget exceeded
        self.router.accumulated_cost_usd = 15.0
        query = "Complex analytical multi-hop query requiring high tier model"
        decision = self.router.route(query)
        self.assertFalse(decision.is_frontier)
        self.assertEqual(decision.selected_model, "ollama/llama3")
        self.assertIn("Daily budget ceiling", decision.reasoning)

if __name__ == "__main__":
    unittest.main()
