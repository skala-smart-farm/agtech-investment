"""그래프 전체가 공유하는 State. 세 묶음으로 나눈다.

① 작업 페이로드: 에이전트가 만든 결과. 다음 에이전트·보고서가 읽는다.
② 제어 메타: Supervisor 라우팅·재작업·종료 판단에 필요한 최소 상태.
③ 추적·복구: 외부 기록(run_log·LangSmith)과 잇는 run_id, 마지막 오류, 결정 로그.

- 누적해야 하는 키에만 Reducer 를 붙이고 나머지는 덮어쓴다.
- 키 끝 주석 형식: '설명  [쓰는 노드 → 읽는 노드]'
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


def add_unique(a: list | None, b: list | None) -> list:
    out = list(a or [])
    for x in b or []:
        if x not in out:
            out.append(x)
    return out


View = Literal["founder", "tech", "market", "competition"]
ViewStatus = Literal[
    "pending",       # 아직 조사하지 않음 (또는 선행 관점이 바뀌어 다시 조사해야 함)
    "collected",     # 조사 완료, Supervisor 검토 전
    "insufficient",  # Supervisor 가 근거 부족으로 재조사 요청
    "sufficient",    # Supervisor 가 근거 충분으로 승인
    "failed",        # 에이전트 실행 실패
    "skipped",       # 결론이 이미 정해져 조사를 건너뜀 (Deal-killer·창업자 기준 미달 확정)
]


class ViewProgress(TypedDict):
    status: ViewStatus
    attempts: int              # 이 후보에서 해당 관점을 실행한 횟수
    missing: list[str]         # 재조사 때 보완할 항목 (Supervisor 가 만든 검색 질문)
    source_ids: list[str]      # Supervisor 가 승인할 때 쓴 근거 id
    exhausted: bool            # 재조사해도 미확인이 줄지 않아 더 재조사하지 않는 관점


class ReportEvaluation(TypedDict):
    passed: bool
    groundedness: bool
    neutrality: bool
    bias_control: bool
    retry: bool                # 미달이고 재작성 상한 전이면 True → report 로 돌아간다
    rule_failures: list[str]   # 규칙 검사(1안)에서 걸린 항목
    judge_reasons: dict        # LLM Judge(2안)의 항목별 이유
    feedback: str              # 미달이면 보고서 재작성에 넘기는 고칠 점


class InvestmentState(TypedDict, total=False):
    # ── ① 작업 페이로드
    domain: str                                        # 도메인 이름  [app → 실행 기록]
    run_date: str                                      # 평가 기준일 YYYY-MM-DD (최근 기간 판정·근거 조회일)  [app → 전 노드]
    registry: Annotated[dict, merge_dict]              # 근거 저장소 {근거 id: 문서·웹 메타데이터}, id → REFERENCE  [전 에이전트 → 전 에이전트·report]
    raw_candidates: list[dict]                         # 이번 라운드 발굴 후보 (이름, 분야, 발굴 채널, 근거 id)  [discover.collect → discover.screen]
    screened: Annotated[list[dict], operator.add]      # 적격성 관문 판정 기록 전체 (통과·탈락 사유)  [discover.screen → report]
    current: dict | None                               # 평가 대상 프로필, None = 후보 소진  [discover.pick·exhausted → 분석 에이전트·decide]
    founder: dict                                      # 창업 시점·인물·마일스톤·창업자 기준 판정  [founder → competition·decide·report]
    tech: dict                                         # 제품·핵심 기술·장단점·차별점 주장·제품/기술력 판정  [tech → market·competition·decide·report]
    market: dict                                       # 시장 규모·성장·수요·시장성 판정  [market → decide·report]
    competition: dict                                  # 경쟁사 비교표·차별점 검증·경쟁 우위 판정  [competition → decide·report]
    market_cache: Annotated[dict, merge_dict]          # 세부 분야별 시장 분석 캐시 (같은 분야 재분석 방지)  [market → market]
    scorecard: dict                                    # 6개 기준 배수 M·결정·보류 유형·뒤집힘·실사·순위  [decide → report]
    evaluations: Annotated[list[dict], operator.add]   # 후보별 평가 누적 (보고서 비교표·보류 사유의 재료)  [decide → report]
    report: dict                                       # 보고서 모드·파일 경로·쪽수·형식 검사 결과  [report → evaluate·app]

    # ── ② 제어 메타
    discovery_rounds: int                              # 발굴 라운드 수 (max_discovery_rounds 로 제한)  [discover.collect → discover 라우터]
    seen: Annotated[list[str], add_unique]             # 이미 다룬 후보의 정규화 이름 (중복 발굴 방지)  [discover.collect → discover.collect]
    queue: list[dict]                                  # 적격 후보 평가 대기열  [discover.screen·pick → discover]
    iterations: int                                    # 평가를 시작한 후보 수 (max_evaluations 로 제한)  [discover.pick → supervisor]
    decision: str                                      # 이번 후보의 결정: '투자' 또는 '보류'  [decide → supervisor]
    supervisor_next: str                               # Supervisor 가 고른 다음 노드  [supervisor → 조건부 간선·worker]
    supervisor_task: dict                              # 재조사 지시 {agent, reason, queries}  [supervisor → 담당 worker]
    view_progress: dict                                # 관점별 진행 {View: ViewProgress} (후보마다 초기화)  [supervisor ⇄ supervisor]
    supervisor_reworks: int                            # 이 후보의 재조사 횟수 (supervisor.max_reworks_per_candidate 로 제한)  [supervisor → supervisor]
    supervisor_steps: int                              # Supervisor 판단 횟수 (supervisor.max_steps 로 제한, 종료 보장)  [supervisor → supervisor]
    supervisor_reviews: dict                           # 후보별 근거 충분성 판단과 사유  [supervisor → app(run_log)]
    supervisor_excluded: list[dict]                    # 근거 부족·실행 실패로 제외한 후보와 사유  [supervisor → report·app]
    supervisor_final_review: dict                      # 보고서 작성 승인 사유  [supervisor → report·app]
    report_revisions: int                              # 품질 평가 미달로 다시 쓴 횟수 (evaluation.max_revisions 로 제한)  [evaluate → evaluate]
    report_evaluation: dict | None                     # 품질 평가 결과 ReportEvaluation  [evaluate → report·app]
    end_reason: str                                    # 종료 사유 invest_found·max_evaluations·exhausted·no_eligible·step_limit  [discover·decide·supervisor → report]

    # ── ③ 추적·복구
    run_id: str                                        # 실행 id. run_log·LangSmith metadata·checkpoint thread_id 와 같은 값  [app → 전 노드]
    last_error: str | None                             # 마지막 worker 실행 오류 (Supervisor 가 처리하면 비움)  [worker → supervisor]
    rag_traces: Annotated[list[dict], operator.add]    # Agentic RAG 수행 기록 (재작성·웹 보완 여부)  [discover·tech·market → app(run_log)]
    log: Annotated[list[str], operator.add]            # 결정과 사유가 담긴 짧은 진행 기록  [전 노드 → app(run_log)]


class DiscoveryIn(TypedDict, total=False):
    """탐색 서브그래프 입력. 누적 키를 넣지 않아 부모 Reducer 가 같은 기록을 두 번 붙이지 않는다."""
    domain: str
    run_date: str
    registry: Annotated[dict, merge_dict]
    discovery_rounds: int
    seen: Annotated[list[str], add_unique]
    queue: list[dict]
    iterations: int


class DiscoveryOut(TypedDict, total=False):
    """탐색 서브그래프 출력."""
    registry: Annotated[dict, merge_dict]
    discovery_rounds: int
    raw_candidates: list[dict]
    seen: Annotated[list[str], add_unique]
    screened: Annotated[list[dict], operator.add]
    queue: list[dict]
    current: dict | None
    iterations: int
    founder: dict
    tech: dict
    market: dict
    competition: dict
    scorecard: dict
    end_reason: str
    rag_traces: Annotated[list[dict], operator.add]
    log: Annotated[list[str], operator.add]
