"""메인 그래프.

    START → discover ─(평가 대상 있음)→ founder → tech → market → competition → decide
               │                                                                 ├─(평가 상한 전)→ discover
               └─(후보 소진)→ report → END                                        └─(평가 상한 도달)→ report → END
"""
from __future__ import annotations

import importlib

from langgraph.graph import END, START, StateGraph

from graph.routes import route_after_decide, route_after_discover
from graph.state import InvestmentState

NODE_NAMES = ("discover", "founder", "tech", "market", "competition", "decide", "report")

_AGENT_NODES = {
    "founder": ("agents.founder", "founder_node"),              # 창업자 (30%)
    "tech": ("agents.tech", "tech_node"),                       # 제품/기술력 (15%)
    "market": ("agents.market", "market_node"),                 # 시장성 (25%)
    "competition": ("agents.competition", "competition_node"),  # 경쟁 우위 (10%)
    "decide": ("agents.decision", "decision_node"),             # 실적·투자조건 (20%) + 투자 결정
    "report": ("agents.report", "report_node"),                 # 보고서
}


def _default_node(name: str):
    if name == "discover":  # 탐색 서브그래프
        from graph.discovery_graph import build_discovery_graph

        return build_discovery_graph()
    module, fn = _AGENT_NODES[name]
    return getattr(importlib.import_module(module), fn)


def build_graph(nodes: dict | None = None):
    """메인 그래프를 컴파일한다. nodes={'decide': 가짜 함수, …} 로 일부 노드를 바꿔 끼울 수 있다(테스트용)."""
    nodes = dict(nodes or {})
    unknown = set(nodes) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"알 수 없는 노드 이름: {sorted(unknown)} (가능: {NODE_NAMES})")

    g = StateGraph(InvestmentState)
    for name in NODE_NAMES:
        g.add_node(name, nodes[name] if name in nodes else _default_node(name))

    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", route_after_discover, {"founder": "founder", "report": "report"})
    g.add_edge("founder", "tech")
    g.add_edge("tech", "market")
    g.add_edge("market", "competition")
    g.add_edge("competition", "decide")
    g.add_conditional_edges("decide", route_after_decide, {"report": "report", "discover": "discover"})
    g.add_edge("report", END)
    return g.compile()
