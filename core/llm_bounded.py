"""구조화 출력 LLM — 출력 상한과 반복 퇴행 1회 재시도.

temperature 0 에서 드물게 같은 토큰을 끝없이 반복하는 경우가 있어, 출력 상한에 닿으면
'반복 금지' 문구와 frequency_penalty 를 붙여 한 번 더 묻는다. 퇴행한 프롬프트는 degenerate.json 에 기록해
다음 실행에서는 바로 재시도 설정을 쓴다.
"""
from __future__ import annotations

import hashlib
import json

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError

from core.config import get_config, path
from core.cost import TRACKER
from core.llm import get_llm

REPEAT_HINT = ("\n\n(출력 규칙: 직전 답이 같은 근거 id·같은 문구를 끝없이 반복해 길이 한도에 닿았다. "
               "근거 id 는 항목마다 한 번만 붙이고, 같은 내용을 되풀이하지 말고 간결하게 끝내라.)")


REPEAT_PENALTY = 0.4


def _registry_path():
    return path(f"{get_config().cache.dir}/degenerate.json")


def _key(schema, prompt) -> str:
    text = prompt if isinstance(prompt, str) else json.dumps([getattr(m, "content", str(m)) for m in prompt], ensure_ascii=False)
    return hashlib.sha256(f"{schema.__name__}\n{text}".encode("utf-8")).hexdigest()[:24]


def _known() -> set[str]:
    p = _registry_path()
    return set(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else set()


def _remember(key: str) -> None:
    p = _registry_path()
    p.write_text(json.dumps(sorted(_known() | {key}), indent=0), encoding="utf-8")


def _with_hint(prompt):
    if isinstance(prompt, str):
        return prompt + REPEAT_HINT
    if isinstance(prompt, list) and prompt and isinstance(prompt[-1], BaseMessage):
        return [*prompt, HumanMessage(REPEAT_HINT.strip())]
    return prompt


class Bounded:
    """structured() 와 같은 .invoke(prompt) 를 제공한다."""

    def __init__(self, schema, role: str, kind: str):
        self.schema, self.role, self.kind = schema, role, kind

    def _llm(self, penalty: float = 0.0):
        get_llm(self.role)  # 캐시·--offline 설정 적용
        cfg = get_config()
        caps = cfg.models.max_output_tokens
        return ChatOpenAI(model=cfg.models[self.role], temperature=cfg.models.temperature, max_retries=3, timeout=120,
                          max_tokens=caps.get(self.kind, caps["default"]), callbacks=[TRACKER],
                          **({"frequency_penalty": penalty} if penalty else {})
                          ).with_structured_output(self.schema, method="json_schema", strict=True)

    def _retry(self, prompt):
        return self._llm(REPEAT_PENALTY).invoke(_with_hint(prompt))

    def invoke(self, prompt):
        key = _key(self.schema, prompt)
        if key in _known():
            return self._retry(prompt)
        try:
            return self._llm().invoke(prompt)
        except LengthFinishReasonError:
            print(f"[LLM] {self.schema.__name__}: 출력 길이 한도 도달(반복 퇴행) → 반복 금지 문구·반복 벌점으로 한 번 더")
            out = self._retry(prompt)
            _remember(key)
            return out
        except RuntimeError as e:
            if "--offline" not in str(e):
                raise
            out = self._retry(prompt)
            _remember(key)
            return out


def bounded(schema, role: str = "generator", kind: str = "default") -> Bounded:
    """kind: config models.max_output_tokens 의 키 (default | report)."""
    return Bounded(schema, role, kind)
