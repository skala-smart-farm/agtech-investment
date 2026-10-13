"""탐색 서브그래프: 다음 적격 후보 1곳을 정한다.

    (진입 · screen 뒤) ─┬─(대기열 있음)──────────────→ pick ─────→ (평가 대상 1곳)
                        ├─(비었고 발굴 라운드 남음)─→ collect → screen → (진입과 같은 분기로 다시)
                        └─(비었고 라운드 소진)───────→ exhausted → (후보 소진)
"""
from __future__ import annotations

import functools
from typing import Callable

from langgraph.graph import END, START, StateGraph

from core.config import run_date
from core.recency import future_round
from graph.routes import route_discovery_entry
from graph.state import DiscoveryIn, DiscoveryOut, InvestmentState

_BRANCHES = {"pick": "pick", "collect": "collect", "exhausted": "exhausted"}


def pick_node(state: dict) -> dict:
    """대기열 맨 앞 후보를 꺼내 평가 대상으로 정하고, 이전 후보의 분석 결과를 비운다."""
    queue = list(state.get("queue", []))
    current = queue.pop(0)
    n = state.get("iterations", 0) + 1
    msg = f"[평가 {n}] {current['official_name']} ({current['region']}, {current['stage']}, {current['segment_id']})"
    print(msg)
    return {"current": current, "queue": queue, "iterations": n, "founder": {}, "tech": {}, "market": {},
            "competition": {}, "scorecard": {}, "log": [msg]}


def exhausted_node(state: dict) -> dict:
    """대기열이 비었고 발굴 라운드도 다 썼다. 평가한 후보가 없으면 no_eligible, 있으면 exhausted."""
    n, rounds = state.get("iterations", 0), state.get("discovery_rounds", 0)
    if n == 0:
        reason, msg = "no_eligible", f"[탐색 종료] 발굴 {rounds}라운드에서 적격 후보 없음 → 보고서"
    else:
        reason, msg = "exhausted", f"[탐색 종료] 적격 후보 {n}곳 평가 후 대기열이 비고 발굴 {rounds}라운드 소진 → 보고서"
    print(msg)
    return {"current": None, "end_reason": reason, "log": [msg]}


def recency_guard(screen_fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """관문 G2(최근 투자)를 평가 기준일로 다시 확인해, 기준일 이후·예정 라운드로 통과한 후보를 탈락시킨다."""
    @functools.wraps(screen_fn)
    def screen(state: dict) -> dict:
        out = screen_fn(state)
        recs = out.get("screened", [])
        if not any(r.get("eligible") for r in recs):
            return out
        date = state.get("run_date") or run_date()
        why = {i: w for i, r in enumerate(recs) if r.get("eligible") and (w := future_round(r, date))}
        if not why:
            return out
        names = {recs[i]["name"] for i in why}
        msg = f"[적격성 보정] G2 기준일 이후·예정 라운드 {len(names)}곳 탈락: {', '.join(sorted(names))}"
        print(msg)
        fixed = {**out, "log": [*out.get("log", []), msg],
                 "screened": [{**r, "eligible": False, "reason": f"G2 {why[i]}"} if i in why else r
                              for i, r in enumerate(recs)]}
        if "queue" in out:
            old = len(state.get("queue") or [])  # 새로 붙은 후보만 거른다
            fixed["queue"] = out["queue"][:old] + [c for c in out["queue"][old:] if c.get("name") not in names]
        return fixed
    return screen


def build_discovery_graph(collect_fn: Callable[[dict], dict] | None = None,
                          screen_fn: Callable[[dict], dict] | None = None):
    """탐색 서브그래프를 컴파일한다. 테스트는 collect_fn·screen_fn 에 가짜 함수를 넣는다."""
    if collect_fn is None:
        from agents.discovery import discovery_node as collect_fn
    if screen_fn is None:
        from agents.eligibility import eligibility_node as screen_fn

    g = StateGraph(InvestmentState, input_schema=DiscoveryIn, output_schema=DiscoveryOut)
    g.add_node("collect", collect_fn)      # 발굴: 채널 검색 → 후보 추출
    g.add_node("screen", recency_guard(screen_fn))  # 적격성 관문 G1~G6
    g.add_node("pick", pick_node)          # 다음 후보 1곳 선택, 후보별 분석 키 초기화
    g.add_node("exhausted", exhausted_node)  # 후보 소진: current 없음, 종료 사유 기록

    g.add_conditional_edges(START, route_discovery_entry, _BRANCHES)
    g.add_edge("collect", "screen")
    g.add_conditional_edges("screen", route_discovery_entry, _BRANCHES)
    g.add_edge("pick", END)
    g.add_edge("exhausted", END)
    return g.compile()
