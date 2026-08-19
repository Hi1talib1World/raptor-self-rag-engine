import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RAPTOR.LLMProvider")

class BaseLLMProvider(ABC):
    """Abstract Base Class for Local and Cloud LLM Providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate full response text."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Yield response tokens for streaming."""
        pass

class OllamaLLMProvider(BaseLLMProvider):
    """Ollama Local-First Zero-Cost LLM Provider (llama3, mistral, phi3, gemma)."""

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "system": system_prompt or ""
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama server connection failed ({e}). Falling back to local offline mode.")
            return f"[Ollama Fallback ({self.model_name})]: Answer grounded on context: {prompt[:150]}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        full_text = self.generate(prompt, system_prompt)
        words = full_text.split(" ")
        for i, w in enumerate(words):
            token = w if i == len(words) - 1 else w + " "
            yield token

class vLLMProvider(BaseLLMProvider):
    """vLLM OpenAI-Compatible High-Throughput Local Server Provider."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", base_url: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a grounded RAG 2.0 assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"vLLM server connection failed ({e}). Returning fallback response.")
            return f"[vLLM Fallback ({self.model_name})]: {prompt[:150]}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        full_text = self.generate(prompt, system_prompt)
        for token in full_text.split(" "):
            yield token + " "

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI Cloud API Provider."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return f"[OpenAI ({self.model_name})]: {prompt[:150]}"
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self.model_name, openai_api_key=self.api_key, temperature=0.0)
            res = llm.invoke(prompt)
            return res.content.strip()
        except Exception:
            return f"[OpenAI Fallback]: {prompt[:150]}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        full = self.generate(prompt, system_prompt)
        for t in full.split(" "):
            yield t + " "

class ClaudeLLMProvider(BaseLLMProvider):
    """Anthropic Claude API Provider (claude-3-5-sonnet, claude-3-opus)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return f"[Claude ({self.model_name})]: {prompt[:150]}"
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=self.model_name, anthropic_api_key=self.api_key, temperature=0.0)
            res = llm.invoke(prompt)
            return res.content.strip()
        except Exception:
            return f"[Claude Fallback]: {prompt[:150]}"

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        full = self.generate(prompt, system_prompt)
        for t in full.split(" "):
            yield t + " "

class MockLLMProvider(BaseLLMProvider):
    """Deterministic Fast Offline Mock LLM Provider for unit tests & offline run."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if "Context:" in prompt:
            context_part = prompt.split("Context:")[1].split("User")[0].strip()
            first_line = context_part.split("\n")[0] if context_part else "retrieved context"
            return f"Based on the retrieved context: {first_line[:150]}"
        return "Based on the retrieved context: The platform supports MQTT, OPC-UA, and Modbus TCP/RTU protocols."

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        full = self.generate(prompt, system_prompt)
        for word in full.split(" "):
            yield word + " "

def get_llm_provider(provider_type: str = "mock", model_name: str = "llama3") -> BaseLLMProvider:
    ptype = provider_type.lower()
    if ptype == "ollama":
        return OllamaLLMProvider(model_name=model_name)
    elif ptype in ["vllm", "sglang"]:
        return vLLMProvider(model_name=model_name)
    elif ptype == "openai":
        return OpenAILLMProvider(model_name=model_name)
    elif ptype in ["claude", "anthropic"]:
        return ClaudeLLMProvider(model_name=model_name)
    else:
        return MockLLMProvider()
