"""분기 함수. State 와 config 만 읽고 다음 노드 이름을 돌려준다."""
from __future__ import annotations

from typing import Literal

from core.config import get_config


def route_discovery_entry(state: dict) -> Literal["pick", "collect", "exhausted"]:
    """탐색 서브그래프 진입과 관문(screen) 뒤에 같은 규칙을 쓴다."""
    if state.get("queue"):
        return "pick"
    if state.get("discovery_rounds", 0) < get_config().workflow.max_discovery_rounds:
        return "collect"
    return "exhausted"


def route_after_discover(state: dict) -> Literal["founder", "report"]:
    """current 가 없으면 후보 소진."""
    return "founder" if state.get("current") else "report"


def stops_on_invest(wf) -> bool:
    """투자 추천에서 평가를 멈추는지. 기본은 멈추지 않는다(발굴 순서가 결론을 정하지 않도록)."""
    return wf.get("stop_on_invest", False) and not wf.get("calibrate", False)


def route_after_decide(state: dict) -> Literal["report", "discover"]:
    wf = get_config().workflow
    if state.get("decision") == "투자" and stops_on_invest(wf):
        return "report"
    if state.get("iterations", 0) >= wf.max_evaluations:  # 반복 상한 (무한 루프 방지)
        return "report"
    return "discover"
