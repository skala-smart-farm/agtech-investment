"""관점별 동적 위임 → 근거 충분성 검토 → 재조사 또는 투자 판단·보고서 승인."""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import get_config
from core.llm_bounded import bounded
from core.prompts import render
from graph.routes import route_after_decide
from graph.state import InvestmentState

Worker = Literal["founder", "tech", "market", "competition"]
ANALYSIS_DEPENDENCIES = {
    "founder": set(), "tech": set(), "market": {"tech"}, "competition": {"founder", "tech"},
}
# 하위 에이전트는 Supervisor가 전달한 입력만 받는다. 다른 에이전트의 State를 직접 읽지 않는다.
WORKER_INPUTS = {
    "discover": {"discovery_rounds", "seen", "queue", "iterations"},
    "founder": {"current"},
    "tech": {"current"},
    "market": {"current", "tech", "market_cache"},
    "competition": {"current", "founder", "tech"},
    "decide": {"current", "founder", "tech", "market", "competition", "iterations"},
    "report": set(InvestmentState.__annotations__),
}


class SupervisorChoice(BaseModel):
    next_agent: Worker
    reason: str = Field(description="이 작업을 먼저 수행할 이유를 한국어 한 문장으로 설명")


class EvidenceReview(BaseModel):
    sufficient: bool = Field(description="조건부 투자 검토 또는 보류의 근거로 충분한가. 미확인을 사실로 취급하지 않는다")
    reason: str = Field(description="충분성 판단 이유. 부족한 관점·주장·증거를 한국어로 구체적으로 적는다")
    evidence_ids: list[str] = Field(description="충분성 판단에 사용한 실재하는 근거 ID")
    rework_agent: Worker | None = Field(description="부족하면 재조사할 분석 에이전트. 충분하면 null")
    queries: list[str] = Field(description="부족하면 새로운 구체적인 검색 질문 1~3개. 충분하면 빈 배열")


def _ask(schema, template: str, context: dict):
    return bounded(schema, kind="supervisor").invoke([
        SystemMessage(render(template)), HumanMessage(json.dumps(context, ensure_ascii=False)),
    ])


def _context(state: dict, completed: list[str]) -> dict:
    c = state["current"]
    reg = state.get("registry", {})
    ids = list(c.get("evidence_ids", []))
    views = {}
    for name in completed:
        result = state.get(name, {})
        ids += result.get("pool_ids", []) + result.get("evidence_ids", [])
        # 문항별 인용·미확인 상태를 자르지 않고 전달한다. 긴 서술만 줄인다.
        views[name] = {"criterion": result.get("criterion", {}),
                       "evidence_ids": result.get("evidence_ids", []),
                       "summary": json.dumps({k: v for k, v in result.items()
                                              if k not in {"criterion", "pool_ids"}}, ensure_ascii=False)[:1800]}
    sources = {i: {k: s.get(k) for k in ("title", "url", "date", "kind", "publisher")}
               | {"text": (s.get("body") or s.get("text") or s.get("snippet") or "")[:900]}
               for i in dict.fromkeys(ids) if (s := reg.get(i))}
    return {"candidate": c, "completed": completed, "analysis": views, "sources": sources,
            "reworks": state.get("supervisor_reworks", 0),
            "previous_queries": state.get("supervisor_queries", [])}


def _dispatch(state: dict, selected: str, completed: list[str], reason: str, **updates) -> dict:
    forwarded = {"domain", "run_date", "registry"} | WORKER_INPUTS.get(selected, set())
    merged = {**state, **updates}
    inbox = {k: merged[k] for k in forwarded if k in merged}
    inbox["supervisor_task"] = updates.get("supervisor_task", {})
    name = (state.get("current") or {}).get("official_name", "후보 없음")
    msg = f"[Supervisor] {name}: {state.get('supervisor_next') or 'START'} → {selected} · {reason}"
    if inbox["supervisor_task"].get("queries"):
        msg += f" · 추가 검색 {inbox['supervisor_task']['queries']}"
    print(msg)
    return {**updates, "supervisor_next": selected, "supervisor_completed": completed,
            "supervisor_task": inbox["supervisor_task"], "supervisor_inbox": inbox, "log": [msg]}


def _finish(state: dict, completed: list[str], selected: str) -> dict:
    names = [e["name"] for e in state.get("evaluations", [])]
    reviews = state.get("supervisor_reviews", {})
    sufficient = all(reviews.get(name, {}).get("sufficient") for name in names)
    final = {"sufficient": sufficient, "reviewed_candidates": names,
             "reason": "평가한 모든 후보의 관점·근거 검토 완료" if sufficient else "근거 검토를 통과하지 않은 후보가 있음"}
    if not names:
        final["reason"] = "적격 평가 후보 없음 — 투자 추천 없이 탐색 결과만 보고"
    return _dispatch(state, selected if sufficient else "halt", completed, final["reason"],
                     supervisor_final_review=final)


def supervisor_node(state: dict) -> dict:
    previous = state.get("supervisor_next")
    completed = list(state.get("supervisor_completed", []))
    if previous == "decide":
        return _finish(state, completed, route_after_decide(state))
    if not state.get("current"):
        if previous == "discover":
            return _finish(state, [], "report")
        return _dispatch(state, "discover", [], "적격 후보 발굴")
    if previous == "discover":
        completed = []
        state = {**state, "supervisor_reworks": 0, "supervisor_queries": []}
    elif previous in ANALYSIS_DEPENDENCIES and previous not in completed:
        completed.append(previous)

    done = set(completed)
    ready = [n for n, deps in ANALYSIS_DEPENDENCIES.items() if n not in done and deps <= done]
    context = _context(state, completed)
    reset = {"supervisor_reworks": state.get("supervisor_reworks", 0),
             "supervisor_queries": state.get("supervisor_queries", [])}
    if ready:
        if len(ready) == 1:
            selected, reason = ready[0], "미수집 관점 중 선행 분석이 완료된 작업"
        else:
            result = _ask(SupervisorChoice, "supervisor", {**context, "available_workers": ready})
            selected, reason = result.next_agent, result.reason
            if selected not in ready:
                raise ValueError(f"Supervisor가 실행 불가능한 작업을 선택했습니다: {selected} (가능: {ready})")
        return _dispatch(state, selected, completed, reason, **reset)

    # 네 관점의 수집 완료만으로 보고서를 허용하지 않는다. 근거를 검토한 뒤 분기한다.
    review = _ask(EvidenceReview, "supervisor_review", context)
    name = state["current"]["official_name"]
    record = {**review.model_dump(), **reset}
    reviews = {**state.get("supervisor_reviews", {}), name: record}
    if review.sufficient:
        cited = set(review.evidence_ids)
        if not cited or not cited <= set(context["sources"]):
            raise ValueError("Supervisor의 충분성 판단에 실재하는 근거 ID가 필요합니다")
        if review.rework_agent is not None or review.queries:
            raise ValueError("충분성 승인과 재작업 요청을 동시에 할 수 없습니다")
        return _dispatch(state, "decide", completed, review.reason, supervisor_reviews=reviews, **reset)

    limit = get_config().supervisor.max_reworks_per_candidate
    if reset["supervisor_reworks"] >= limit:
        final = {"sufficient": False, "reason": f"재조사 한도 {limit}회 소진: {review.reason}", "candidate": name}
        return _dispatch(state, "halt", completed, final["reason"], supervisor_reviews=reviews,
                         supervisor_final_review=final, **reset)
    target = review.rework_agent
    queries = list(dict.fromkeys(q.strip() for q in review.queries if q.strip()))
    if target not in ANALYSIS_DEPENDENCIES or not 1 <= len(queries) <= 3:
        raise ValueError("근거 부족이면 담당 에이전트와 구체적인 검색 질문 1~3개가 필요합니다")
    if set(queries) & set(reset["supervisor_queries"]):
        raise ValueError("재조사는 이전 질문을 반복하지 않고 새로운 검색 질문을 사용해야 합니다")
    # 선행 관점이 바뀌면 그 결과를 사용한 분석도 다시 수행한다.
    invalid = {target}
    for worker, deps in ANALYSIS_DEPENDENCIES.items():
        if deps & invalid:
            invalid.add(worker)
    completed = [n for n in completed if n not in invalid]
    reset["supervisor_reworks"] += 1
    reset["supervisor_queries"] += queries
    return _dispatch(state, target, completed, f"근거 부족 → 재작업: {review.reason}",
                     **{n: {} for n in invalid}, supervisor_reviews=reviews,
                     supervisor_task={"agent": target, "reason": review.reason, "queries": queries}, **reset)
