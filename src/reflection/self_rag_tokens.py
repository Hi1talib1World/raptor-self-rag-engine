from typing import List, Tuple

try:
    from pydantic import BaseModel
    class SelfRAGOutput(BaseModel):
        answer: str
        should_retrieve: bool
        is_relevant: bool
        is_supported: bool
        utility_score: float
        reflection_tokens: List[str]
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class SelfRAGOutput:
        answer: str
        should_retrieve: bool
        is_relevant: bool
        is_supported: bool
        utility_score: float
        reflection_tokens: List[str] = field(default_factory=list)

class SelfRAGReflectionEngine:
    """Self-RAG Reflection Token & Critique Engine."""

    TOKENS = {
        "RETRIEVE_YES": "[Retrieve: YES]",
        "RETRIEVE_NO": "[Retrieve: NO]",
        "IS_REL_YES": "[IsRel: YES]",
        "IS_REL_NO": "[IsRel: NO]",
        "IS_SUP_FULL": "[IsSup: FULL]",
        "IS_SUP_PARTIAL": "[IsSup: PARTIAL]",
        "IS_SUP_NONE": "[IsSup: NONE]",
        "IS_USE_HIGH": "[IsUse: 5/5]",
        "IS_USE_LOW": "[IsUse: 1/5]",
    }

    def evaluate_retrieval_need(self, query: str) -> Tuple[bool, str]:
        q_lower = query.lower()
        if any(w in q_lower for w in ["hello", "hi", "who are you", "what can you do"]):
            return False, self.TOKENS["RETRIEVE_NO"]
        return True, self.TOKENS["RETRIEVE_YES"]

    def evaluate_relevance(self, query: str, context: str) -> Tuple[bool, str]:
        if not context.strip():
            return False, self.TOKENS["IS_REL_NO"]
        q_words = set(query.lower().split())
        c_words = set(context.lower().split())
        overlap = len(q_words.intersection(c_words)) / (len(q_words) or 1)
        return (True, self.TOKENS["IS_REL_YES"]) if overlap >= 0.15 else (False, self.TOKENS["IS_REL_NO"])

    def evaluate_grounding_and_utility(self, generated_text: str, context: str) -> Tuple[bool, float, List[str]]:
        tokens = []
        if not context.strip():
            tokens.extend([self.TOKENS["IS_SUP_NONE"], self.TOKENS["IS_USE_LOW"]])
            return False, 0.2, tokens

        a_words = set(generated_text.lower().split())
        c_words = set(context.lower().split())
        grounded_ratio = len(a_words.intersection(c_words)) / (len(a_words) or 1)

        if grounded_ratio >= 0.4:
            tokens.extend([self.TOKENS["IS_SUP_FULL"], self.TOKENS["IS_USE_HIGH"]])
            return True, 1.0, tokens
        elif grounded_ratio >= 0.2:
            tokens.extend([self.TOKENS["IS_SUP_PARTIAL"], self.TOKENS["IS_USE_HIGH"]])
            return True, 0.7, tokens
        else:
            tokens.extend([self.TOKENS["IS_SUP_NONE"], self.TOKENS["IS_USE_LOW"]])
            return False, 0.2, tokens

    def run_critique_loop(self, query: str, context: str, generator_func) -> SelfRAGOutput:
        tokens = []
        should_ret, ret_token = self.evaluate_retrieval_need(query)
        tokens.append(ret_token)

        is_rel, rel_token = self.evaluate_relevance(query, context)
        tokens.append(rel_token)

        raw_answer = generator_func(query, context)
        is_sup, utility, sup_tokens = self.evaluate_grounding_and_utility(raw_answer, context)
        tokens.extend(sup_tokens)

        tagged_answer = f"{raw_answer}\n\n[Self-RAG Reflection: {' '.join(tokens)}]"
        return SelfRAGOutput(
            answer=tagged_answer,
            should_retrieve=should_ret,
            is_relevant=is_rel,
            is_supported=is_sup,
            utility_score=utility,
            reflection_tokens=tokens
        )
