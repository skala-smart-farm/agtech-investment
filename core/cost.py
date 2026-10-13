"""LLM 토큰 사용량·비용 집계. 캐시에서 꺼낸 응답은 비용에 넣지 않는다."""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler

from core.config import get_config


def model_price(model: str) -> tuple[float, float]:
    """모델 이름 → (입력, 출력) USD / 100만 토큰 (config.yaml models.price_per_mtok)."""
    models = get_config().models
    table = models.get("price_per_mtok") or {}
    hit = max((k for k in table if model.startswith(k)), key=len, default=None)
    pi, po = table[hit] if hit else table[models.generator]
    return float(pi), float(po)


class CostTracker(BaseCallbackHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.cached = self.input_tokens = self.output_tokens = 0
        self.usd = 0.0

    def on_llm_end(self, response, **kwargs) -> None:
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        with self.lock:
            if not out:
                self.cached += 1
                return
            model = str(out.get("model_name", ""))
            pi, po = model_price(model)
            i, o = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            ci = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            self.calls += 1
            self.input_tokens += i
            self.output_tokens += o
            self.usd += (i - ci) / 1e6 * pi + ci / 1e6 * pi / 4 + o / 1e6 * po

    def summary(self) -> dict:
        return {"api_calls": self.calls, "cache_hits": self.cached, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4)}


TRACKER = CostTracker()
