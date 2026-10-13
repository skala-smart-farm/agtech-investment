"""관점별 동적 위임 → 근거 충분성 검토 → 재조사 또는 투자 판단·보고서 승인.

관점 진행 상태는 State 의 view_progress 하나로 관리한다 (pending → collected → sufficient / insufficient / failed).
"""
from __future__ import annotations

import json
from copy import deepcopy
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
VIEWS = tuple(ANALYSIS_DEPENDENCIES)
DONE = ("collected", "sufficient")
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
        SystemMessage(render(template, domain=get_config().domain.name)),
        HumanMessage(json.dumps(context, ensure_ascii=False)),
    ])


def _fresh_progress() -> dict:
    return {v: {"status": "pending", "attempts": 0, "missing": [], "source_ids": []} for v in VIEWS}


def _view_ids(state: dict, view: str) -> list[str]:
    result = state.get(view) or {}
    return list(dict.fromkeys(result.get("pool_ids", []) + result.get("evidence_ids", [])))


def _context(state: dict, progress: dict) -> dict:
    c = state["current"]
    cfg = get_config().supervisor
    reg = state.get("registry", {})
    completed = [v for v in VIEWS if progress[v]["status"] in DONE]
    ids = list(c.get("evidence_ids", []))
    views = {}
    for name in completed:
        result = state.get(name, {})
        ids += _view_ids(state, name)
        # 문항별 인용·미확인 상태를 자르지 않고 전달한다. 긴 서술만 줄인다.
        views[name] = {"criterion": result.get("criterion", {}),
                       "evidence_ids": result.get("evidence_ids", []),
                       "summary": json.dumps({k: v for k, v in result.items()
                                              if k not in {"criterion", "pool_ids"}}, ensure_ascii=False)[:cfg.context_summary_chars]}
    sources = {i: {k: s.get(k) for k in ("title", "url", "date", "kind", "publisher")}
               | {"text": (s.get("body") or s.get("text") or s.get("snippet") or "")[:cfg.context_source_chars]}
               for i in dict.fromkeys(ids) if (s := reg.get(i))}
    return {"candidate": c, "completed": completed, "analysis": views, "sources": sources,
            "reworks": state.get("supervisor_reworks", 0),
            "previous_queries": [q for v in VIEWS for q in progress[v]["missing"]]}


def _dispatch(state: dict, selected: str, reason: str, progress: dict | None = None, **updates) -> dict:
    """다음 노드와 사유를 State 에 남긴다. 분석 관점으로 보내면 그 관점의 실행 횟수를 올린다."""
    task = updates.pop("supervisor_task", {})
    name = (state.get("current") or {}).get("official_name", "후보 없음")
    msg = f"[Supervisor] {name}: {state.get('supervisor_next') or 'START'} → {selected} · {reason}"
    if task.get("queries"):
        msg += f" · 추가 검색 {task['queries']}"
    print(msg)
    out = {**updates, "supervisor_next": selected, "supervisor_task": task, "log": [msg]}
    if progress is not None:
        if selected in progress:
            progress[selected] = {**progress[selected], "attempts": progress[selected]["attempts"] + 1}
        out["view_progress"] = progress
    return out


def _finish(state: dict, selected: str, **updates) -> dict:
    names = [e["name"] for e in state.get("evaluations", [])]
    excluded = [x["candidate"] for x in updates.get("supervisor_excluded", state.get("supervisor_excluded", []))]
    reason = "평가한 모든 후보의 관점·근거 검토 완료" if names else "적격 평가 후보 없음 — 투자 추천 없이 탐색 결과만 보고"
    if updates.get("end_reason") == "step_limit":
        reason = f"Supervisor 판단 상한 {get_config().supervisor.max_steps}회 도달 — 지금까지 평가한 후보로 보고"
    if excluded:
        reason += f" (근거 부족으로 제외 {len(excluded)}곳: {', '.join(excluded)})"
    final = {"sufficient": True, "reviewed_candidates": names, "excluded_candidates": excluded, "reason": reason}
    return _dispatch(state, selected, reason, supervisor_final_review=final, **updates)


def _new_queries(review: EvidenceReview, previous: set[str]) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in review.queries if q.strip() and q.strip() not in previous))[:3]


def _normalize(review: EvidenceReview, sources: dict) -> EvidenceReview:
    """LLM 답의 형식 오류는 실행을 멈추지 않고 보정한다. 실재하지 않는 근거 ID 로는 승인하지 않는다."""
    cited = [i for i in review.evidence_ids if i in sources]
    if review.sufficient and not cited:
        return review.model_copy(update={"sufficient": False, "evidence_ids": [],
                                         "reason": f"승인 근거 ID 가 실재하지 않음: {review.reason}"})
    if review.sufficient:
        return review.model_copy(update={"evidence_ids": cited, "rework_agent": None, "queries": []})
    target = review.rework_agent if review.rework_agent in ANALYSIS_DEPENDENCIES else None
    return review.model_copy(update={"evidence_ids": cited, "rework_agent": target})


def _exclude(state: dict, name: str, reason: str, progress: dict, **updates) -> dict:
    """재조사로도 근거가 부족하거나 실행에 실패한 후보는 제외하고 다음 후보로 넘어간다 (실행 전체를 멈추지 않는다)."""
    excluded = [*state.get("supervisor_excluded", []), {"candidate": name, "reason": reason}]
    updates = {**updates, "supervisor_excluded": excluded, "view_progress": progress}
    if state.get("iterations", 0) >= get_config().workflow.max_evaluations:
        return _finish(state, "report", end_reason="max_evaluations", **updates)
    return _dispatch(state, "discover", f"근거 부족으로 제외 → 다음 후보: {reason}", **updates)


def supervisor_node(state: dict) -> dict:
    cfg = get_config().supervisor
    steps = state.get("supervisor_steps", 0) + 1
    previous = state.get("supervisor_next")
    progress = deepcopy(state.get("view_progress") or _fresh_progress())
    if steps > cfg.max_steps:  # 종료 보장: 판단 횟수 상한에 닿으면 지금까지 평가한 후보로 보고서
        return _finish(state, "report", end_reason="step_limit", supervisor_steps=steps)
    if previous == "decide":
        return _finish(state, route_after_decide(state), supervisor_steps=steps)
    if not state.get("current"):
        if previous == "discover":
            return _finish(state, "report", supervisor_steps=steps)
        return _dispatch(state, "discover", "적격 후보 발굴", supervisor_steps=steps)
    if previous == "discover":  # 새 후보: 관점 진행 상태와 재조사 횟수를 초기화
        progress = _fresh_progress()
        state = {**state, "supervisor_reworks": 0}
    base = {"supervisor_steps": steps, "supervisor_reworks": state.get("supervisor_reworks", 0)}
    name = state["current"]["official_name"]
    if previous in VIEWS:
        if state.get("last_error"):
            progress[previous] = {**progress[previous], "status": "failed"}
            return _exclude(state, name, f"{previous} 에이전트 실행 실패: {state['last_error']}", progress,
                            last_error=None, **base)
        progress[previous] = {**progress[previous], "status": "collected"}

    done = {v for v in VIEWS if progress[v]["status"] in DONE}
    ready = [n for n, deps in ANALYSIS_DEPENDENCIES.items() if n not in done and deps <= done]
    context = _context(state, progress)
    if ready:
        if len(ready) == 1:
            selected, reason = ready[0], "미수집 관점 중 선행 분석이 완료된 작업"
        else:
            result = _ask(SupervisorChoice, "supervisor", {**context, "available_workers": ready})
            selected, reason = result.next_agent, result.reason
            if selected not in ready:  # 실행할 수 없는 작업을 고르면 가능한 첫 작업으로
                selected, reason = ready[0], f"LLM 선택({selected})이 실행 불가능해 가능한 작업으로 대체"
        return _dispatch(state, selected, reason, progress, **base)

    # 네 관점의 수집 완료만으로 보고서를 허용하지 않는다. 근거를 검토한 뒤 분기한다.
    limit = cfg.max_reworks_per_candidate
    previous_queries = set(context["previous_queries"])
    review = _normalize(_ask(EvidenceReview, "supervisor_review", context), context["sources"])
    queries = _new_queries(review, previous_queries)
    if not review.sufficient and base["supervisor_reworks"] < limit and not (queries and review.rework_agent):
        review = _normalize(_ask(EvidenceReview, "supervisor_review", {
            **context, "feedback": ("승인하려면 sources 에 있는 근거 ID 를 적어라. 부족하면 rework_agent 와 "
                                    "previous_queries 를 제외한 새로운 검색 질문을 제안하라."),
        }), context["sources"])
        queries = _new_queries(review, previous_queries)
    reviews = {**state.get("supervisor_reviews", {}), name: {**review.model_dump(), **base}}
    if review.sufficient:
        cited = set(review.evidence_ids)
        for v in VIEWS:
            progress[v] = {**progress[v], "status": "sufficient",
                           "source_ids": [i for i in _view_ids(state, v) if i in cited]}
        return _dispatch(state, "decide", review.reason, progress, supervisor_reviews=reviews, **base)

    if base["supervisor_reworks"] >= limit:
        return _exclude(state, name, f"재조사 한도 {limit}회 소진: {review.reason}", progress,
                        supervisor_reviews=reviews, **base)
    target = review.rework_agent
    if not queries or target is None:
        return _exclude(state, name, f"재조사 대상·질문 생성 실패: {review.reason}", progress,
                        supervisor_reviews=reviews, **base)
    # 선행 관점이 바뀌면 그 결과를 사용한 분석도 다시 수행한다.
    invalid = {target}
    for worker, deps in ANALYSIS_DEPENDENCIES.items():
        if deps & invalid:
            invalid.add(worker)
    progress[target] = {**progress[target], "status": "insufficient", "missing": progress[target]["missing"] + queries}
    for v in invalid - {target}:
        progress[v] = {**progress[v], "status": "pending"}
    base["supervisor_reworks"] += 1
    return _dispatch(state, target, f"근거 부족 → 재작업: {review.reason}", progress,
                     **{n: {} for n in invalid}, supervisor_reviews=reviews,
                     supervisor_task={"agent": target, "reason": review.reason, "queries": queries}, **base)
