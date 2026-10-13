"""Supervisor 그래프.

    START → supervisor ⇄ discover · founder · tech · market · competition · decide
                     └→ report → evaluate ─(미달, 재작성 상한 전)→ report
                                         └─(통과 또는 상한)→ END

근거가 부족하거나 실행에 실패한 후보는 Supervisor 가 제외하고 다음 후보로 넘어간다.
"""
from __future__ import annotations

from copy import deepcopy

from langgraph.graph import END, START, StateGraph

from agents.evaluator import evaluate_node, route_after_evaluate
from agents.supervisor import ANALYSIS_DEPENDENCIES, WORKER_INPUTS, supervisor_node
from core.config import get_config
from graph.builder import NODE_NAMES, _default_node
from graph.state import InvestmentState

RECURSION_SLACK = 20  # 탐색 서브그래프 진입·보고서 등 Supervisor 판단 밖의 단계 여유분


def worker_node(name: str, node):
    """하위 에이전트는 Supervisor 가 정한 입력(WORKER_INPUTS)만 복사해 받고, 결과만 돌려준다."""
    def run(state, config):
        forwarded = {"domain", "run_date", "registry", "run_id"} | WORKER_INPUTS.get(name, set())
        inbox = deepcopy({k: state[k] for k in forwarded if k in state})
        task = state.get("supervisor_task") or {}
        if name == "market" and state.get("supervisor_reworks", 0):
            inbox["market_cache"] = {}  # 기술 재조사로 시장 분석이 무효화된 경우도 다시 분석한다.
        if task.get("agent") == name and task.get("queries"):
            from tools.sources import SourceRegistry
            from tools.web_search import web_search

            reg = SourceRegistry(inbox.get("registry"))
            ids = []
            for query in task["queries"]:
                ids += web_search(query, reg, name, recent=False, deep=True, raw=True)
            inbox["registry"] = reg.data
            inbox["current"]["evidence_ids"] = list(dict.fromkeys(inbox["current"].get("evidence_ids", []) + ids))
            inbox["supervisor_research_ids"] = ids
        if name not in ANALYSIS_DEPENDENCIES:
            out = node.invoke(inbox, config) if hasattr(node, "invoke") else node(inbox)
            if name == "report":
                out["report"] = {**out["report"], "evidence_review": state.get("supervisor_final_review")}
            return out
        try:  # 분석 에이전트 실패는 실행을 멈추지 않고 last_error 로 Supervisor 에 알린다 (그 후보만 제외)
            out = node.invoke(inbox, config) if hasattr(node, "invoke") else node(inbox)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:300]
            print(f"[{name}] 실행 실패 → Supervisor 에 보고: {msg}")
            return {"last_error": msg, "log": [f"[{name}] 실행 실패: {msg}"]}
        if not out.get(name):
            return {"last_error": "분석 결과를 돌려주지 않음"}
        return out
    return run


def supervisor_recursion_limit() -> int:
    """종료 보장은 supervisor.max_steps 가 먼저 한다. recursion_limit 은 그보다 넉넉한 최후 안전장치다.
    Supervisor 판단 1회 = supervisor + worker 2스텝, 품질 평가 루프 1회 = report + evaluate 2스텝."""
    cfg = get_config()
    steps = 2 * cfg.supervisor.max_steps + 2 * (cfg.evaluation.max_revisions + 1) + RECURSION_SLACK
    return max(cfg.workflow.recursion_limit, steps)


def build_supervisor_graph(nodes: dict | None = None, checkpointer=None):
    """nodes 로 일부 노드를 가짜 함수로 바꿔 끼울 수 있다(테스트용). checkpointer 를 주면 run_id 를 thread_id 로 재개할 수 있다."""
    nodes = dict(nodes or {})
    unknown = set(nodes) - {*NODE_NAMES, "supervisor", "evaluate"}
    if unknown:
        raise ValueError(f"알 수 없는 노드 이름: {sorted(unknown)}")
    g = StateGraph(InvestmentState)
    g.add_node("supervisor", nodes.get("supervisor", supervisor_node))
    for name in NODE_NAMES:
        node = nodes[name] if name in nodes else _default_node(name)
        g.add_node(name, worker_node(name, node))
        g.add_edge(name, "evaluate" if name == "report" else "supervisor")
    g.add_node("evaluate", nodes.get("evaluate", evaluate_node))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda state: state["supervisor_next"],
                            {name: name for name in NODE_NAMES})
    g.add_conditional_edges("evaluate", route_after_evaluate, {"report": "report", "end": END})
    return g.compile(checkpointer=checkpointer)
