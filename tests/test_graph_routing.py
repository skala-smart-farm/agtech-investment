"""메인 그래프·분기·실행 옵션 검사. 키·네트워크 없이 모든 에이전트 노드를 가짜 함수로 바꿔 끝까지 돌린다.

- 분기 함수(graph/routes.py)의 경우별 결과
- 컴파일된 그래프의 간선 == 설계서 DESIGN_EDGES (Graph(안) + 창업자, END 로 바로 가는 보정 전용 간선 없음, 병렬 없음)
- 끝까지 실행: 모두 보류 → 소진(B안) / 첫 후보 투자(A안) / 평가 상한 / 적격 후보 없음(C안) / 보정 실행
- app.py 의 --calibrate·--threshold·--out 이 설정과 run_log 에 반영되는지
"""
from __future__ import annotations

import json

import pytest
from langgraph.graph import END, START

from core.config import get_config
from graph.builder import NODE_NAMES, build_graph
from graph.discovery_graph import build_discovery_graph
from graph.routes import stops_on_invest, route_after_decide, route_after_discover, route_discovery_entry

# 설계서 D.2 그림(Graph(안) + 👤 창업자)과 1:1. 보정 실행도 report 노드에서 끝나므로 discover·decide 에서 END 로 가는 간선은 없다
DESIGN_EDGES = {(START, "discover"), ("discover", "founder"), ("discover", "report"), ("founder", "tech"),
                ("tech", "market"), ("market", "competition"), ("competition", "decide"), ("decide", "report"),
                ("decide", "discover"), ("report", END)}
ANALYSIS = ("founder", "tech", "market", "competition")


@pytest.fixture
def supervisor_llm(monkeypatch):
    """LLM이 tech → market → founder → competition 순서를 고르게 한다. 네트워크 호출 없음."""
    from types import SimpleNamespace
    from agents import supervisor

    fake = SimpleNamespace(prompts=[], reviews=[], searches=[], order=["tech", "market", "competition", "founder"])

    def invoke(schema, messages):
        context = json.loads(messages[-1].content)
        fake.prompts.append(context)
        if schema is supervisor.EvidenceReview:
            if fake.reviews:
                return fake.reviews.pop(0)
            return schema(sufficient=True, reason="각 관점의 검증 가능한 근거가 있음", evidence_ids=["Wabc12"],
                          rework_agent=None, queries=[])
        ready = context["available_workers"]
        selected = next(name for name in fake.order if name in ready)
        return supervisor.SupervisorChoice(next_agent=selected, reason="제품과 시장부터 확인")

    monkeypatch.setattr(supervisor, "bounded", lambda schema, **kw: SimpleNamespace(
        invoke=lambda messages: invoke(schema, messages)))

    def search(query, reg, agent, **kw):
        fake.searches.append((agent, query))
        sid = f"W{len(fake.searches):05x}"
        reg.data[sid] = {"kind": "web", "snippet": "추가 검증 결과", "title": "재조사 근거", "url": "https://example.org/new"}
        return [sid]

    import tools.web_search
    monkeypatch.setattr(tools.web_search, "web_search", search)
    return fake


def supervisor_nodes(fakes, inputs=None):
    """기존 가짜 노드에 인용 근거를 더해 충분성 검토와 Supervisor의 입력 경계를 검사한다."""
    nodes = fakes.nodes()
    for name in ANALYSIS:
        node = nodes[name]

        def run(state, name=name, node=node):
            if inputs is not None:
                inputs.append((name, state))
            out = node(state)
            ids = ["Wabc12", *state.get("supervisor_research_ids", [])]
            out[name].update(evidence_ids=ids, pool_ids=ids, criterion={"rows": [
                {"qid": name, "answer": "YES", "quote": "검증된 사실", "evidence_ids": ["Wabc12"]}]})
            out["registry"] = {**state.get("registry", {}), "Wabc12": {
                "kind": "web", "snippet": "검증된 사실", "title": "검증 근거", "url": "https://example.org/evidence"}}
            return out

        nodes[name] = run
    return nodes


@pytest.mark.parametrize("names, cap, stop, calibrate, expected, end_reason", [
    ([], 10, False, False, [], "no_eligible"),
    (["A", "B"], 10, False, False, ["A", "B"], "exhausted"),
    ([str(i) for i in range(11)], 10, False, False, [str(i) for i in range(10)], "max_evaluations"),
    (["A", "B"], 10, True, False, ["A"], "invest_found"),
    (["A", "B"], 10, True, True, ["A", "B"], "exhausted"),
])
def test_supervisor_runs_to_completion(supervisor_llm, set_cfg, names, cap, stop, calibrate, expected, end_reason):
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    set_cfg("workflow.max_evaluations", cap)
    set_cfg("workflow.stop_on_invest", stop)
    set_cfg("workflow.calibrate", calibrate)
    f = Fakes([names], decisions={name: "투자" for name in names})
    state = {"registry": {}, "discovery_rounds": 0, "iterations": 0, "queue": [], "seen": [],
             "evaluations": [], "screened": [], "log": [], "rag_traces": []}
    inputs = []
    out = build_supervisor_graph(supervisor_nodes(f, inputs)).invoke(state, {"recursion_limit": supervisor_recursion_limit()})
    assert [e["name"] for e in out.get("evaluations", [])] == expected
    assert out["end_reason"] == end_reason
    assert f.calls.count("report") == 1 and not f.stale
    analysis_calls = [name for name in f.calls if name in {*ANALYSIS, "decide"}]
    assert analysis_calls == ["tech", "market", "founder", "competition", "decide"] * len(expected)
    for prompt in supervisor_llm.prompts:
        if "available_workers" not in prompt:
            assert set(prompt["completed"]) == set(ANALYSIS)
            assert prompt["sources"] and all(v["criterion"]["rows"] for v in prompt["analysis"].values())
            continue
        done = set(prompt["completed"])
        ready = set(prompt["available_workers"])
        assert not done & ready
        assert "market" not in ready or "tech" in done
        assert "competition" not in ready or {"founder", "tech"} <= done
    assert len(out.get("screened", [])) == len(names)
    assert out["supervisor_final_review"]["sufficient"]
    assert set(out.get("supervisor_reviews", {})) == set(expected)
    for name, inbox in inputs:
        assert "supervisor_inbox" not in inbox and "evaluations" not in inbox
        if name in {"founder", "tech"}:
            assert not set(ANALYSIS) & inbox.keys()
        if name == "market":
            assert "tech" in inbox and not {"founder", "competition"} & inbox.keys()
    if calibrate:
        assert out["report"]["mode"] == "calibrate"


def test_supervisor_falls_back_on_invalid_llm_choice(monkeypatch):
    from types import SimpleNamespace
    from agents import supervisor

    monkeypatch.setattr(supervisor, "bounded", lambda *a, **kw: SimpleNamespace(
        invoke=lambda _: supervisor.SupervisorChoice(next_agent="competition", reason="경쟁사부터 비교")))
    out = supervisor.supervisor_node({"current": {"official_name": "A"}, "supervisor_next": "discover"})
    assert out["supervisor_next"] == "founder" and "실행 불가능" in out["log"][0]


def test_supervisor_edges():
    from graph.supervisor import build_supervisor_graph

    edges = build_supervisor_graph(Fakes([]).nodes()).get_graph().edges
    expected = {(START, "supervisor"), ("report", "evaluate"), ("evaluate", "report"), ("evaluate", END)}
    expected |= {("supervisor", name) for name in NODE_NAMES}
    expected |= {(name, "supervisor") for name in NODE_NAMES if name != "report"}
    assert {(e.source, e.target) for e in edges} == expected
    with pytest.raises(ValueError):
        build_supervisor_graph({"verify": lambda _: {}})


@pytest.mark.parametrize("target", ANALYSIS)
def test_supervisor_reworks_insufficient_view(supervisor_llm, target):
    from agents.supervisor import EvidenceReview
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    supervisor_llm.reviews.append(EvidenceReview(
        sufficient=False, reason="핵심 주장의 제3자 근거 부족", evidence_ids=["Wabc12"],
        rework_agent=target, queries=["A 유료 고객 설치 실적 검증"]))
    f = Fakes([["A"]])
    inputs = []
    out = build_supervisor_graph(supervisor_nodes(f, inputs)).invoke(
        {}, {"recursion_limit": supervisor_recursion_limit()})
    assert f.calls.count(target) == 2
    assert f.calls.count("decide") == f.calls.count("report") == 1
    assert supervisor_llm.searches == [(target, "A 유료 고객 설치 실적 검증")]
    retried = [inbox for name, inbox in inputs if name == target][-1]
    assert retried["supervisor_research_ids"] == ["W00001"]
    assert "W00001" in retried["current"]["evidence_ids"]
    assert out[target]["pool_ids"] == ["Wabc12", "W00001"]
    assert out["supervisor_reviews"]["A"]["supervisor_reworks"] == 1
    assert out["supervisor_final_review"]["sufficient"]
    reviews = [p for p in supervisor_llm.prompts if "available_workers" not in p]
    assert len(reviews) == 2 and "W00001" in reviews[-1]["sources"]
    if target == "tech":
        assert f.calls.count("market") == f.calls.count("competition") == 2
        assert [s for name, s in inputs if name == "market"][-1].get("market_cache") == {}
    if target == "founder":
        assert f.calls.count("competition") == 2


def test_supervisor_holds_candidate_when_rework_exhausted(supervisor_llm, set_cfg):
    from agents.supervisor import EvidenceReview
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    set_cfg("supervisor.max_reworks_per_candidate", 1)
    supervisor_llm.reviews.extend([
        EvidenceReview(sufficient=False, reason="시장 수요 근거 부족", evidence_ids=[],
                       rework_agent="market", queries=["A 시장 수요 실증"]),
        EvidenceReview(sufficient=False, reason="추가 검색 후에도 시장 수요 확인 불가", evidence_ids=[],
                       rework_agent="market", queries=["A 시장 수요 실측"]),
    ])
    f = Fakes([["A", "B"]])
    nodes = supervisor_nodes(f)
    shortfalls = []
    real_decide = nodes["decide"]
    nodes["decide"] = lambda s: shortfalls.append(s.get("evidence_shortfall")) or real_decide(s)
    out = build_supervisor_graph(nodes).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    assert f.calls.count("market") == 3  # A 2회(재조사 포함) + B 1회
    assert f.calls.count("decide") == 2 and f.calls.count("report") == 1  # A 도 빼지 않고 채점한다
    assert [e["name"] for e in out["evaluations"]] == ["A", "B"] and not out.get("supervisor_excluded")
    assert shortfalls[0].startswith("재조사 한도 1회 소진") and shortfalls[1] is None  # A 만 근거 부족 보류로 채점
    assert out["supervisor_reviews"]["A"]["reason"] == "추가 검색 후에도 시장 수요 확인 불가"


def test_supervisor_does_not_approve_with_fabricated_evidence(supervisor_llm):
    from agents.supervisor import EvidenceReview
    from graph.supervisor import build_supervisor_graph

    supervisor_llm.reviews.append(EvidenceReview(sufficient=True, reason="충분", evidence_ids=["Wfake99"],
                                                rework_agent=None, queries=[]))
    f = Fakes([["A"]])
    out = build_supervisor_graph(supervisor_nodes(f)).invoke({}, {"recursion_limit": 100})
    reviews = [p for p in supervisor_llm.prompts if "available_workers" not in p]
    assert len(reviews) == 2 and reviews[-1].get("feedback")  # 가짜 ID 승인은 무효 → 다시 물어 실재 ID 로 승인
    assert out["supervisor_reviews"]["A"]["evidence_ids"] == ["Wabc12"]
    assert f.calls.count("decide") == f.calls.count("report") == 1


@pytest.mark.parametrize("new_queries, expected_searches, sufficient", [
    (["A 고객 확인", "A 매출 확인"], ["A 고객 확인", "A 매출 확인"], True),
    (["A 고객 확인"], ["A 고객 확인", "A 매출 확인"], True),
    (["A 고객 확인"], ["A 고객 확인"], False),
])
def test_supervisor_handles_repeated_rework_queries(supervisor_llm, new_queries, expected_searches, sufficient):
    from agents.supervisor import EvidenceReview
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    def shortage(queries):
        return EvidenceReview(sufficient=False, reason="추가 검증 필요", evidence_ids=[],
                              rework_agent="market", queries=queries)

    supervisor_llm.reviews.extend([shortage(["A 고객 확인"]), shortage(new_queries)])
    if new_queries == ["A 고객 확인"]:
        supervisor_llm.reviews.append(shortage(["A 매출 확인"] if sufficient else ["A 고객 확인"]))
    f = Fakes([["A"]])
    out = build_supervisor_graph(supervisor_nodes(f)).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    assert [query for _, query in supervisor_llm.searches] == expected_searches
    assert f.calls.count("decide") == 1 and f.calls.count("report") == 1 and not out.get("supervisor_excluded")
    assert out["supervisor_reviews"]["A"]["sufficient"] is sufficient
    assert any(p.get("feedback") for p in supervisor_llm.prompts) == (new_queries == ["A 고객 확인"])


def test_supervisor_worker_input_is_isolated():
    from graph.supervisor import worker_node

    state = {"current": {"official_name": "A"}, "registry": {"Wabc12": {"snippet": "원문"}}, "market": {"hidden": True},
             "view_progress": {}, "supervisor_steps": 3}

    def node(inbox):
        assert "market" not in inbox and "view_progress" not in inbox  # 다른 관점 결과·제어 상태는 받지 않는다
        inbox["current"]["official_name"] = "변경"
        inbox["registry"]["Wabc12"]["snippet"] = "변경"
        return {"founder": {"name": "A"}}

    worker_node("founder", node)(state, {})
    assert state["current"]["official_name"] == "A" and state["registry"]["Wabc12"]["snippet"] == "원문"


def test_supervisor_worker_failure_is_reported_not_raised():
    from graph.supervisor import worker_node

    def boom(inbox):
        raise RuntimeError("API 한도 초과")

    out = worker_node("tech", boom)({"current": {"official_name": "A"}}, {})
    assert out["last_error"] == "RuntimeError: API 한도 초과"


def test_supervisor_can_choose_different_order(supervisor_llm):
    from graph.supervisor import build_supervisor_graph

    supervisor_llm.order = ["founder", "tech", "competition", "market"]
    f = Fakes([["A"]])
    build_supervisor_graph(supervisor_nodes(f)).invoke({}, {"recursion_limit": 100})
    assert [n for n in f.calls if n in ANALYSIS] == ["founder", "tech", "competition", "market"]


# ── 분기 함수
@pytest.mark.parametrize("state, calibrate, expected", [
    ({"decision": "투자", "iterations": 1}, False, "discover"),      # 투자 추천이어도 멈추지 않고 다음 후보
    ({"decision": "투자", "iterations": 10}, False, "report"),       # 평가 상한 → 보고서(A안: 통과 후보 중 배수 1위)
    ({"decision": "보류", "iterations": 10}, False, "report"),       # 보류 · 평가 상한 → 보고서(B안)
    ({"decision": "보류", "iterations": 3}, False, "discover"),      # 보류 → 다른 스타트업 탐색
    ({"decision": "투자", "iterations": 3}, True, "discover"),       # 보정 실행: 투자여도 계속
    ({"decision": "투자", "iterations": 10}, True, "report"),        # 보정 실행: 상한에서 report(no-op) → END
    ({"decision": None, "iterations": 3}, True, "discover"),         # 보정 실행에서 결정을 비워 둔 경우
])
def test_route_after_decide(set_cfg, state, calibrate, expected):
    if calibrate:  # app.py --calibrate 와 같은 설정
        set_cfg("workflow.calibrate", True)
        set_cfg("workflow.stop_on_invest", False)
    assert route_after_decide(state) == expected


def test_route_after_decide_stop_on_invest_option(set_cfg):
    """workflow.stop_on_invest=true 면 노션 Graph(안)의 '투자 추천 → 보고서 생성' 그대로 멈춘다."""
    set_cfg("workflow.stop_on_invest", True)
    assert route_after_decide({"decision": "투자", "iterations": 1}) == "report"
    assert route_after_decide({"decision": "보류", "iterations": 1}) == "discover"


def test_route_after_decide_calibrate_never_stops(set_cfg):
    set_cfg("workflow.calibrate", True)
    set_cfg("workflow.stop_on_invest", True)  # 보정 실행은 이 값과 관계없이 멈추지 않는다
    assert route_after_decide({"decision": "투자", "iterations": 3}) == "discover"


@pytest.mark.parametrize("calibrate", [False, True])
def test_route_after_discover(set_cfg, calibrate):
    set_cfg("workflow.calibrate", calibrate)
    assert route_after_discover({"current": {"official_name": "A"}}) == "founder"
    assert route_after_discover({"current": None}) == "report"  # 보정 실행도 report 로 (report 가 no-op)


def test_route_discovery_entry():
    assert get_config().workflow.max_discovery_rounds == 2
    assert route_discovery_entry({"queue": [{"official_name": "A"}], "discovery_rounds": 2}) == "pick"
    assert route_discovery_entry({"queue": [], "discovery_rounds": 1}) == "collect"
    assert route_discovery_entry({"queue": [], "discovery_rounds": 2}) == "exhausted"


# ── 가짜 노드
class Fakes:
    """rounds[i] = (i+1)라운드 발굴 후보, decisions = 후보별 결정(없으면 보류). calls 에 노드 실행 순서를 남긴다."""

    def __init__(self, rounds: list[list[str]], decisions: dict[str, str | None] | None = None,
                 eligible: set[str] | None = None):
        self.rounds, self.decisions, self.eligible = rounds, decisions or {}, eligible
        self.calls: list[str] = []
        self.stale: list[str] = []  # pick 이 초기화하지 않아 이전 후보 결과가 남아 있던 경우

    def collect(self, s):
        self.calls.append("collect")
        r = s.get("discovery_rounds", 0) + 1
        names = [n for n in (self.rounds[r - 1] if r <= len(self.rounds) else []) if n not in s.get("seen", [])]
        return {"raw_candidates": [{"name": n} for n in names], "discovery_rounds": r, "seen": names,
                "registry": {f"W{r}": {}}, "log": [f"[발굴 {r}라운드] {len(names)}곳"]}

    def screen(self, s):
        self.calls.append("screen")
        recs = [{"name": c["name"], "eligible": self.eligible is None or c["name"] in self.eligible}
                for c in s.get("raw_candidates", [])]
        passed = [{"official_name": r["name"], "region": "KR", "stage": "Seed", "segment_id": "robotics"}
                  for r in recs if r["eligible"]]
        return {"screened": recs, "queue": list(s.get("queue", [])) + passed,
                "log": [f"[적격성 검증 {s.get('discovery_rounds')}라운드] 통과 {len(passed)}곳"]}

    def analysis(self, key):
        def node(s):
            self.calls.append(key)
            if s.get(key):
                self.stale.append(key)
            name = s["current"]["official_name"]
            return {key: {"name": name}, "log": [f"[{key}] {name}"]}
        return node

    def decide(self, s):
        """계약 C2·P3 명세대로 end_reason 을 조건부로 남긴다(투자 ∧ stop_on_invest → invest_found, 보류 ∧ 상한 → max_evaluations)."""
        self.calls.append("decide")
        wf = get_config().workflow
        name = s["current"]["official_name"]
        d = self.decisions.get(name, "보류")
        out = {"scorecard": {"name": name}, "decision": d, "evaluations": [{"name": name, "decision": d}],
               "log": [f"[decide] {name} {d}"]}
        if d == "투자" and stops_on_invest(wf):
            out["end_reason"] = "invest_found"
        elif s["iterations"] >= wf.max_evaluations:
            out["end_reason"] = "max_evaluations"
        return out

    def report(self, s):
        """보정 모드에서는 렌더링하지 않는다(P5 계약). 그 밖에는 받은 종료 사유와 설정을 돌려준다."""
        self.calls.append("report")
        cfg = get_config()
        if cfg.workflow.get("calibrate"):
            return {"report": {"mode": "calibrate"}, "log": ["[report] 보정 실행: 생략"]}
        return {"report": {"mode": "invest" if any(e["decision"] == "투자" for e in s.get("evaluations", [])) else "hold",
                           "end_reason": s.get("end_reason"), "scenario": cfg.report.scenario,
                           "threshold": cfg.decision.threshold, "pdf": "fake.pdf"},
                "log": ["[report]"]}

    def nodes(self) -> dict:
        return {"discover": build_discovery_graph(self.collect, self.screen),
                **{k: self.analysis(k) for k in ANALYSIS}, "decide": self.decide, "report": self.report}

    def run(self) -> dict:
        state = {"registry": {}, "discovery_rounds": 0, "iterations": 0, "queue": [], "seen": [],
                 "evaluations": [], "screened": [], "log": [], "rag_traces": []}
        return build_graph(self.nodes()).invoke(state, {"recursion_limit": get_config().workflow.recursion_limit})


CANDIDATE = ["founder", "tech", "market", "competition", "decide"]  # 후보 1곳의 순차 흐름 (병렬 없음)


def _check_no_duplicates(f: Fakes, out: dict) -> None:
    """누적 키에 같은 기록이 두 번 붙지 않았고, 기록 수 == 노드 실행 수(가짜 노드 + pick + exhausted)."""
    assert len(out["log"]) == len(set(out["log"]))
    picks = sum(line.startswith("[평가 ") for line in out["log"])
    exhausted = sum(line.startswith("[탐색 종료]") for line in out["log"])
    assert picks == out["iterations"]
    assert len(out["log"]) == len(f.calls) + picks + exhausted
    names = [r["name"] for r in out["screened"]]
    assert len(names) == len(set(names))
    assert f.stale == []  # 후보가 바뀔 때마다 분석 결과가 비워졌다


# ── 그래프 구조
def test_edges_match_design():
    edges = build_graph(Fakes([]).nodes()).get_graph().edges
    assert {(e.source, e.target) for e in edges} == DESIGN_EDGES
    assert not {("discover", END), ("decide", END)} & {(e.source, e.target) for e in edges}
    # 병렬 없음: 조건 없는 간선은 노드마다 하나 이하, 조건 간선과 섞이지 않는다
    for n in NODE_NAMES:
        plain = [e for e in edges if e.source == n and not e.conditional]
        cond = [e for e in edges if e.source == n and e.conditional]
        assert len(plain) <= 1 and not (plain and cond), n


def test_default_graph_edges_match_design():
    """실제 에이전트 노드로 컴파일한 그래프(app.py 가 쓰는 것)도 같은 간선이다."""
    app = build_graph()
    assert {(e.source, e.target) for e in app.get_graph().edges} == DESIGN_EDGES
    assert set(app.get_graph().nodes) == {START, END, *NODE_NAMES}


def test_unknown_node_name_rejected():
    with pytest.raises(ValueError):
        build_graph({"verify": lambda s: {}})


# ── 끝까지 실행
def test_run_all_hold_loops_back_then_exhausted():
    """후보 3곳 모두 보류 → 투자 판단 뒤 탐색으로 3번 돌아감(2라운드 발굴 포함) → 소진 → report 1회."""
    f = Fakes([["A", "B"], ["C"]])
    out = f.run()
    assert f.calls == ["collect", "screen", *CANDIDATE, *CANDIDATE, "collect", "screen", *CANDIDATE, "report"]
    assert out["iterations"] == 3 and out["end_reason"] == "exhausted" and out["current"] is None
    assert [e["name"] for e in out["evaluations"]] == ["A", "B", "C"]
    assert out["report"]["mode"] == "hold" and out["report"]["end_reason"] == "exhausted"
    _check_no_duplicates(f, out)


def test_run_default_continues_after_invest():
    """기본(stop_on_invest=false): 첫 후보가 투자여도 나머지를 평가한 뒤 보고서로 간다."""
    f = Fakes([["A", "B", "C"]], decisions={"A": "투자", "C": "투자"})
    out = f.run()
    assert [e["name"] for e in out["evaluations"]] == ["A", "B", "C"] and out["iterations"] == 3
    assert f.calls.count("report") == 1 and out["report"]["mode"] == "invest"
    assert out["end_reason"] != "invest_found"
    _check_no_duplicates(f, out)


def test_run_stop_on_invest_goes_straight_to_report(set_cfg):
    set_cfg("workflow.stop_on_invest", True)  # 노션 Graph(안) 그대로
    f = Fakes([["A", "B"]], decisions={"A": "투자"})
    out = f.run()
    assert f.calls == ["collect", "screen", *CANDIDATE, "report"]
    assert out["iterations"] == 1 and out["end_reason"] == "invest_found"
    assert out["report"]["mode"] == "invest"
    assert [c["official_name"] for c in out["queue"]] == ["B"]  # 미평가 적격 후보는 대기열에 남는다
    _check_no_duplicates(f, out)


def test_run_stops_at_max_evaluations(set_cfg):
    set_cfg("workflow.max_evaluations", 2)
    f = Fakes([["A", "B", "C"]])
    out = f.run()
    assert f.calls == ["collect", "screen", *CANDIDATE, *CANDIDATE, "report"]
    assert out["iterations"] == 2 and out["end_reason"] == "max_evaluations"
    assert [c["official_name"] for c in out["queue"]] == ["C"]
    _check_no_duplicates(f, out)


def test_run_default_cap_within_recursion_limit():
    """기본 설정(평가 상한 10, recursion_limit 100)으로 12곳이 모두 보류여도 상한에서 멈춘다."""
    f = Fakes([[f"C{i:02d}" for i in range(12)]])
    out = f.run()
    assert out["iterations"] == get_config().workflow.max_evaluations == 10
    assert out["end_reason"] == "max_evaluations" and f.calls.count("report") == 1
    _check_no_duplicates(f, out)


def test_run_no_eligible():
    f = Fakes([["A"], ["B"]], eligible=set())
    out = f.run()
    assert f.calls == ["collect", "screen", "collect", "screen", "report"]
    assert out["iterations"] == 0 and out["end_reason"] == "no_eligible" and out["current"] is None
    _check_no_duplicates(f, out)


def test_run_calibrate_continues_after_invest_and_ends_at_report(set_cfg):
    """보정 실행: 투자가 나와도 멈추지 않고 평가 상한까지 평가 → report(렌더링 생략) → END."""
    set_cfg("workflow.calibrate", True)
    set_cfg("workflow.stop_on_invest", False)
    set_cfg("workflow.max_evaluations", 3)
    f = Fakes([["A", "B", "C", "D"]], decisions={"A": "투자", "B": "투자", "C": None})
    out = f.run()
    assert f.calls == ["collect", "screen", *CANDIDATE, *CANDIDATE, *CANDIDATE, "report"]
    assert out["iterations"] == 3 and [e["name"] for e in out["evaluations"]] == ["A", "B", "C"]
    assert out["report"] == {"mode": "calibrate"}
    _check_no_duplicates(f, out)


# ── app.py 실행 옵션
def test_apply_options():
    import app

    cfg = get_config()
    assert app.apply_options(app.parse_args([]), cfg) == "main"
    assert cfg.workflow.stop_on_invest is False and cfg.report.output_dir == "outputs"

    assert app.apply_options(app.parse_args(["--calibrate"]), cfg) == "calibrate"
    assert cfg.workflow.stop_on_invest is False and cfg.workflow.calibrate is True
    assert cfg.report.output_dir == "outputs/calibration"
    get_config.cache_clear()

    cfg = get_config()
    assert app.apply_options(app.parse_args(["--threshold", "1.3", "--out", "outputs/scenario_hold"]), cfg) == "scenario"
    assert cfg.decision.threshold == 1.3 and cfg.report.output_dir == "outputs/scenario_hold"
    assert cfg.report.scenario is True and cfg.workflow.calibrate is False

    with pytest.raises(SystemExit):  # 보정과 시나리오는 같이 쓸 수 없다
        app.parse_args(["--calibrate", "--out", "x"])


@pytest.fixture
def fake_app(monkeypatch, tmp_path):
    """app.main 을 가짜 노드 그래프로 돌리고, 파일은 tmp_path 아래에 쓰게 한다."""
    import app

    def tmp_path_of(rel: str):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    fakes = Fakes([["A", "B", "C"]], decisions={"B": "투자"})
    monkeypatch.setattr(app, "_check_browser", lambda: None)
    monkeypatch.setattr(app, "require_keys", lambda: None)
    monkeypatch.setattr(app, "build_graph", lambda: build_graph(fakes.nodes()))
    monkeypatch.setattr(app, "get_run_date", lambda: "2026-09-30")
    monkeypatch.setattr(app, "path", tmp_path_of)
    return app, fakes, tmp_path


def test_app_scenario_run_writes_run_log(fake_app):
    app, fakes, tmp = fake_app
    app.main(["--sequential", "--threshold", "1.3", "--out", "outputs/scenario_hold"])
    log = json.loads((tmp / "outputs/scenario_hold/run_log.json").read_text(encoding="utf-8"))
    assert log["mode"] == "scenario" and log["threshold"] == 1.3 and log["end_reason"] == "exhausted"
    assert log["report"]["scenario"] is True and log["report"]["threshold"] == 1.3  # 노드가 바뀐 설정을 읽었다
    assert log["evaluated"] == 3 and [e["name"] for e in log["evaluations"]] == ["A", "B", "C"]  # B 가 투자여도 계속
    assert list(log["evaluations"][0]) == ["name", "region", "stage", "decision", "hold_type", "multiplier",
                                           "score100", "reasons", "flip", "roi", "criteria", "rows"]
    assert not (tmp / "outputs/run_log.json").exists()  # 제출 폴더는 건드리지 않는다


def test_app_supervisor_writes_run_log(fake_app, supervisor_llm, monkeypatch):
    from graph import supervisor

    app, fakes, tmp = fake_app
    real_builder = supervisor.build_supervisor_graph
    monkeypatch.setattr(supervisor, "build_supervisor_graph", lambda: real_builder(supervisor_nodes(fakes)))
    checked = []
    monkeypatch.setattr(app, "require_keys", lambda: checked.append(True))
    app.main(["--out", "outputs/supervisor_report"])  # 기본 = Supervisor, 새로 호출
    log = json.loads((tmp / "outputs/supervisor_report/run_log.json").read_text(encoding="utf-8"))
    assert checked == [True]
    assert log["orchestration"] == "supervisor" and log["evaluated"] == 3
    assert any(line.startswith("[Supervisor]") for line in log["log"])
    assert get_config().cache.dir == "cache_fresh"
    assert log["report"]["scenario"] is True
    assert log["supervisor"]["final_review"]["sufficient"]


def test_app_shortfall_candidate_stays_in_reference(fake_app, supervisor_llm, monkeypatch, set_cfg):
    from agents.supervisor import EvidenceReview
    from graph import supervisor

    set_cfg("supervisor.max_reworks_per_candidate", 0)
    supervisor_llm.reviews.append(EvidenceReview(sufficient=False, reason="근거 없음", evidence_ids=[],
                                                rework_agent="tech", queries=["A 제품 실증"]))
    app, fakes, tmp = fake_app
    real_builder = supervisor.build_supervisor_graph
    monkeypatch.setattr(supervisor, "build_supervisor_graph", lambda: real_builder(supervisor_nodes(fakes)))
    monkeypatch.setattr(app, "require_keys", lambda: None)
    calibrated = []
    monkeypatch.setattr(app, "finish_calibration", lambda state, *a: calibrated.append(
        [e["name"] for e in state["evaluations"]]) or {})
    app.main(["--calibrate"])
    log = json.loads((tmp / "outputs/calibration/run_log.json").read_text(encoding="utf-8"))
    assert not log["supervisor"]["excluded"] and not log["supervisor"]["reviews"]["A"]["sufficient"]
    assert calibrated and calibrated[0] == ["A", "B", "C"]  # 근거 부족 후보도 빼지 않고 기준 집단에 들어간다


def test_app_calibrate_writes_reference_class(fake_app, monkeypatch):
    import agents.decision as decision

    app, fakes, tmp = fake_app
    got = {}

    def write_reference_class(evaluations, path, run_date):
        got["ref"] = ([e["name"] for e in evaluations], path, run_date)
        return {"n": len(evaluations), "members": [{"name": e["name"]} for e in evaluations]}

    def write_threshold_sensitivity(ref_path, out_path):
        got["sens"] = (ref_path, out_path)
        return {"invest_count_at": {"1.10": 1}}

    monkeypatch.setattr(decision, "write_reference_class", write_reference_class)
    monkeypatch.setattr(decision, "write_threshold_sensitivity", write_threshold_sensitivity)
    monkeypatch.setattr(app, "v1_evaluated_names", lambda: ["C", "B", "A"])
    app.main(["--sequential", "--calibrate"])

    assert fakes.calls.count("decide") == 3  # B 가 투자여도 멈추지 않고 소진까지
    assert got["ref"] == (["A", "B", "C"], str(tmp / "data/reference_class.json"), "2026-09-30")
    assert got["sens"] == (str(tmp / "data/reference_class.json"), str(tmp / "outputs/eval/threshold_sensitivity.json"))
    log = json.loads((tmp / "outputs/calibration/run_log.json").read_text(encoding="utf-8"))
    assert log["mode"] == "calibrate" and log["end_reason"] == "exhausted" and log["report"] == {"mode": "calibrate"}
    assert log["calibration"]["reference_n"] == 3 and log["calibration"]["same_as_v1_evaluated"] is True
    assert log["calibration"]["reference_enough"] is False  # 3곳 < reference_min_n 5 → 경고, 본 실행은 fallback


def test_v1_members_gate(monkeypatch):
    import app

    names = app.v1_evaluated_names()
    if names is None:
        pytest.skip("v1-safe 태그 없음")
    assert len(names) == 10 and "메타파머스" in names
    assert app.check_v1_members(list(reversed(names))) is True
    assert app.check_v1_members(names[:9] + ["다른 회사"]) is False


def test_supervisor_excludes_candidate_when_agent_returns_nothing(supervisor_llm):
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    f = Fakes([["A", "B"]])
    nodes = supervisor_nodes(f)
    real_tech = nodes["tech"]
    nodes["tech"] = lambda state: {"tech": {}} if state["current"]["official_name"] == "A" else real_tech(state)
    out = build_supervisor_graph(nodes).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    assert [x["candidate"] for x in out["supervisor_excluded"]] == ["A"]
    assert [e["name"] for e in out["evaluations"]] == ["B"] and f.calls.count("report") == 1


def test_supervisor_quality_loop_rewrites_report(supervisor_llm, monkeypatch, set_cfg):
    from agents import evaluator
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    set_cfg("evaluation.max_revisions", 2)
    verdicts = iter([False, True])  # 첫 보고서 미달 → 재작성 → 통과

    def fake_evaluate(state):
        ok = next(verdicts)
        n = state.get("report_revisions", 0)
        return {"report_evaluation": {"passed": ok, "retry": not ok, "feedback": "" if ok else "인용 보강"},
                "report_revisions": n + (not ok)}

    f = Fakes([["A"]])
    nodes = {**supervisor_nodes(f), "evaluate": fake_evaluate}
    out = build_supervisor_graph(nodes).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    assert f.calls.count("report") == 2 and out["report_revisions"] == 1 and out["report_evaluation"]["passed"]


def test_supervisor_step_limit_finishes_with_report(supervisor_llm, set_cfg):
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    set_cfg("supervisor.max_steps", 4)
    f = Fakes([["A", "B"]])
    out = build_supervisor_graph(supervisor_nodes(f)).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    assert out["end_reason"] == "step_limit" and out["supervisor_steps"] == 5
    assert f.calls.count("report") == 1 and "decide" not in f.calls


def test_supervisor_view_progress_tracks_status_and_attempts(supervisor_llm):
    from agents.supervisor import EvidenceReview
    from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

    supervisor_llm.reviews.append(EvidenceReview(sufficient=False, reason="기술 근거 부족", evidence_ids=[],
                                                 rework_agent="tech", queries=["A 실증 결과"]))
    f = Fakes([["A"]])
    out = build_supervisor_graph(supervisor_nodes(f)).invoke({}, {"recursion_limit": supervisor_recursion_limit()})
    vp = out["view_progress"]
    assert all(v["status"] == "sufficient" for v in vp.values())
    assert vp["tech"]["attempts"] == 2 and vp["tech"]["missing"] == ["A 실증 결과"]
    assert vp["market"]["attempts"] == vp["competition"]["attempts"] == 2 and vp["founder"]["attempts"] == 1
    assert vp["tech"]["source_ids"] == ["Wabc12"]
