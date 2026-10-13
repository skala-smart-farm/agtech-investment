"""START → supervisor ⇄ 각 에이전트 → report / 근거 부족 중단 → END."""
from __future__ import annotations

from copy import deepcopy

from langgraph.graph import END, START, StateGraph

from agents.supervisor import ANALYSIS_DEPENDENCIES, supervisor_node
from core.config import get_config
from graph.builder import NODE_NAMES, _default_node
from graph.state import InvestmentState


class SupervisorState(InvestmentState, total=False):
    supervisor_next: str
    supervisor_completed: list[str]
    supervisor_task: dict
    supervisor_inbox: dict
    supervisor_reworks: int
    supervisor_queries: list[str]
    supervisor_reviews: dict
    supervisor_final_review: dict


def worker_node(name: str, node):
    def run(state, config):
        # 각 노드는 Supervisor가 만든 독립 입력을 받아 결과만 Supervisor에 반환한다.
        inbox = deepcopy(state["supervisor_inbox"])
        task = inbox.pop("supervisor_task", {})
        if name == "market" and state.get("supervisor_reworks", 0):
            inbox["market_cache"] = {}  # 기술 재조사로 시장 분석이 무효화된 경우도 다시 분석한다.
        if task.get("queries"):
            from tools.sources import SourceRegistry
            from tools.web_search import web_search

            reg = SourceRegistry(inbox.get("registry"))
            ids = []
            for query in task["queries"]:
                ids += web_search(query, reg, name, recent=False, deep=True, raw=True)
            inbox["registry"] = reg.data
            inbox["current"]["evidence_ids"] = list(dict.fromkeys(
                inbox["current"].get("evidence_ids", []) + ids))
            inbox["supervisor_research_ids"] = ids
        out = node.invoke(inbox, config) if hasattr(node, "invoke") else node(inbox)
        if name in ANALYSIS_DEPENDENCIES and not out.get(name):
            raise ValueError(f"{name} 에이전트가 분석 결과를 반환하지 않았습니다")
        if name == "report":
            out["report"] = {**out["report"], "evidence_review": state["supervisor_final_review"]}
        return out
    return run


def halt_node(state: dict) -> dict:
    reason = state["supervisor_final_review"]["reason"]
    return {"end_reason": "insufficient_evidence", "report": {"mode": "insufficient_evidence", "pdf": None},
            "log": [f"[Supervisor 중단] 보고서 작성 승인 안 됨: {reason}"]}


def supervisor_recursion_limit() -> int:
    cfg = get_config()
    # 재조사 때 의존 분석을 다시 수행하는 최악 경로까지 허용한다. 보고서 승인은 별도의 근거 검토다.
    steps = 2 * cfg.workflow.max_evaluations * (len(NODE_NAMES) + 4 * cfg.supervisor.max_reworks_per_candidate) + 20
    return max(cfg.workflow.recursion_limit, steps)


def build_supervisor_graph(nodes: dict | None = None):
    nodes = dict(nodes or {})
    unknown = set(nodes) - {*NODE_NAMES, "supervisor"}
    if unknown:
        raise ValueError(f"알 수 없는 노드 이름: {sorted(unknown)}")
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", nodes.get("supervisor", supervisor_node))
    for name in NODE_NAMES:
        node = nodes[name] if name in nodes else _default_node(name)
        g.add_node(name, worker_node(name, node))
        g.add_edge(name, END if name == "report" else "supervisor")
    g.add_node("halt", halt_node)
    g.add_edge("halt", END)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda state: state["supervisor_next"],
                            {name: name for name in (*NODE_NAMES, "halt")})
    return g.compile()
