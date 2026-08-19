import math
import time
from typing import List, Optional, Tuple

try:
    from pydantic import BaseModel
    class RoutingDecision(BaseModel):
        target_model: str
        is_frontier_model: bool
        predictive_uncertainty: float
        entropy_score: float
        reasoning: str
        latency_ms: float
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class RoutingDecision:
        target_model: str
        is_frontier_model: bool
        predictive_uncertainty: float
        entropy_score: float
        reasoning: str
        latency_ms: float

class AdaptiveQueryRouter:
    """vLLM / SGLang Predictive Uncertainty & Entropy Adaptive Query Router."""

    def __init__(
        self,
        on_prem_slm: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        frontier_api_model: str = "gpt-4o",
        uncertainty_threshold: float = 0.35
    ):
        self.on_prem_slm = on_prem_slm
        self.frontier_api_model = frontier_api_model
        self.uncertainty_threshold = uncertainty_threshold

    def compute_logprob_uncertainty(self, token_logprobs: List[float]) -> Tuple[float, float]:
        if not token_logprobs:
            return 0.0, 0.0

        probs = [math.exp(lp) for lp in token_logprobs]
        avg_prob = sum(probs) / len(probs)
        entropy = -sum(p * math.log(p + 1e-9) for p in probs) / len(probs)
        uncertainty = max(0.0, min(1.0, 1.0 - avg_prob + (entropy * 0.2)))
        return round(uncertainty, 4), round(entropy, 4)

    def route_query(self, query: str, sample_logprobs: Optional[List[float]] = None) -> RoutingDecision:
        start_t = time.time()
        
        if sample_logprobs:
            uncertainty, entropy = self.compute_logprob_uncertainty(sample_logprobs)
        else:
            q_lower = query.lower()
            uncertainty = 0.5 if any(w in q_lower for w in ["compare", "analyze", "why", "multi-hop"]) else 0.1
            entropy = round(uncertainty * 1.5, 4)

        if uncertainty >= self.uncertainty_threshold:
            target = self.frontier_api_model
            is_frontier = True
            reason = f"Uncertainty ({uncertainty:.2f}) >= {self.uncertainty_threshold} -> Escalated to Frontier Model ({target})"
        else:
            target = self.on_prem_slm
            is_frontier = False
            reason = f"Uncertainty ({uncertainty:.2f}) < {self.uncertainty_threshold} -> Dispatched to On-Prem SLM ({target})"

        latency = round((time.time() - start_t) * 1000.0, 2)
        return RoutingDecision(
            target_model=target,
            is_frontier_model=is_frontier,
            predictive_uncertainty=uncertainty,
            entropy_score=entropy,
            reasoning=reason,
            latency_ms=latency
        )
