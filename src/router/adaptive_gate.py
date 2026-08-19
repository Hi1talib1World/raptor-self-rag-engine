import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("RAPTOR.AdaptiveGate")

class RoutingDecision(BaseModel):
    """Encapsulates model selection, entropy uncertainty, and cost routing rationale."""
    selected_model: str = Field(..., description="Target LLM/SLM model name or endpoint")
    provider: str = Field(..., description="Provider type: 'ollama', 'vllm', 'openai', or 'claude'")
    is_frontier: bool = Field(..., description="True if routed to frontier cloud API; False for local SLM")
    predictive_uncertainty: float = Field(..., description="Predictive logprob uncertainty score in [0.0, 1.0]")
    entropy_score: float = Field(..., description="Shannon entropy score H(Y|X) derived from output token logprobs")
    complexity_score: float = Field(..., description="Estimated query structural complexity in [0.0, 1.0]")
    reasoning: str = Field(..., description="Detailed explanation of routing rationale")
    latency_ms: float = Field(..., description="Routing decision overhead in milliseconds")

class AdaptiveRouterGate:
    """Production Uncertainty-Based Adaptive Token Router.
    Routes simple factual queries to a local Ollama/vLLM endpoint (e.g. Llama-3-8B / Llama-3.2-3B)
    and ambiguous multi-hop analytical queries to a frontier external model (e.g. GPT-4o / Claude 3.5 Sonnet).
    """

    def __init__(
        self,
        local_ollama_model: str = "ollama/llama3",
        frontier_model: str = "gpt-4o",
        local_ollama_base_url: str = "http://localhost:11434",
        uncertainty_threshold: float = 0.35,
        daily_budget_usd: float = 10.0
    ):
        self.local_ollama_model = local_ollama_model
        self.frontier_model = frontier_model
        self.local_ollama_base_url = local_ollama_base_url.rstrip("/")
        self.uncertainty_threshold = uncertainty_threshold
        self.daily_budget_usd = daily_budget_usd
        self.accumulated_cost_usd = 0.0

    def compute_token_entropy_and_uncertainty(
        self,
        query: str,
        token_logprobs: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """Calculate Shannon Entropy, Predictive Uncertainty, and Query Complexity.
        - Shannon Entropy: H = - sum(p * log(p))
        - Predictive Uncertainty: 1.0 - mean(probs) + (0.2 * H)
        - Complexity: Syntactic length + comparative & multi-hop keyword presence
        """
        q_lower = query.lower()
        words = q_lower.split()

        # 1. Structural Complexity
        complexity = 0.1
        if len(words) > 22:
            complexity += 0.25
        
        multi_hop_keywords = ["compare", "contrast", "difference", "versus", "vs", "why", "multi-hop", "analyze", "evaluate", "relationship"]
        if any(kw in q_lower for kw in multi_hop_keywords):
            complexity += 0.4

        # 2. Shannon Entropy & Logprob Uncertainty
        if token_logprobs and len(token_logprobs) > 0:
            probs = [math.exp(lp) for lp in token_logprobs]
            avg_prob = sum(probs) / len(probs)
            # Shannon entropy over token probabilities
            entropy = -sum(p * math.log(p + 1e-9) for p in probs) / len(probs)
            uncertainty = max(0.0, min(1.0, 1.0 - avg_prob + (entropy * 0.2)))
        else:
            uncertainty = min(1.0, complexity * 0.85)
            entropy = round(uncertainty * 1.5, 4)

        return round(uncertainty, 4), round(entropy, 4), round(min(1.0, complexity), 4)

    def route(self, query: str, sample_logprobs: Optional[List[float]] = None) -> RoutingDecision:
        """Synchronously route query based on continuous uncertainty & complexity metrics."""
        start_t = time.time()
        uncertainty, entropy, complexity = self.compute_token_entropy_and_uncertainty(query, sample_logprobs)

        budget_exceeded = self.accumulated_cost_usd >= self.daily_budget_usd

        if budget_exceeded:
            selected = self.local_ollama_model
            provider = "ollama"
            is_frontier = False
            reason = f"Daily budget ceiling (${self.daily_budget_usd}) reached -> Forced local Ollama route ({selected})"
        elif uncertainty >= self.uncertainty_threshold or complexity >= 0.5:
            selected = self.frontier_model
            provider = "openai" if "gpt" in selected.lower() else "claude"
            is_frontier = True
            reason = f"High uncertainty/complexity ({uncertainty:.2f} >= {self.uncertainty_threshold}) -> Dispatched to Frontier Model ({selected})"
        else:
            selected = self.local_ollama_model
            provider = "ollama"
            is_frontier = False
            reason = f"Simple factual query (complexity: {complexity:.2f}, uncertainty: {uncertainty:.2f}) -> Routed to Local Ollama ({selected})"

        latency = round((time.time() - start_t) * 1000.0, 2)
        logger.info(f"[Adaptive Router Gate] Query: '{query[:40]}...' | Model: {selected} | Latency: {latency}ms")

        return RoutingDecision(
            selected_model=selected,
            provider=provider,
            is_frontier=is_frontier,
            predictive_uncertainty=uncertainty,
            entropy_score=entropy,
            complexity_score=complexity,
            reasoning=reason,
            latency_ms=latency
        )

    async def route_async(self, query: str, sample_logprobs: Optional[List[float]] = None) -> RoutingDecision:
        """Asynchronously route query based on token entropy & uncertainty metrics."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.route, query, sample_logprobs)
