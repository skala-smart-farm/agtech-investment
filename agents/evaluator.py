"""보고서 품질 평가 노드 (Hybrid = 1안 규칙 검사 + 2안 LLM Judge). 미달이면 보고서를 다시 쓰게 한다.

평가 항목: Groundedness · 중립성 · 편향 통제.
항목마다 규칙 검사와 LLM Judge 를 둘 다 통과해야 충족이고, 세 항목이 모두 충족이어야 통과다.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from core.config import get_config
from core.judge import company_keys
from core.llm_bounded import bounded
from core.prompts import render
from tools.grounding import norm

ITEMS = ("groundedness", "neutrality", "bias_control")
LABELS = {"groundedness": "Groundedness", "neutrality": "중립성", "bias_control": "편향 통제"}
NUM_CITE = re.compile(r"\[\d+(?:, \d+)*\]")
# 인용 번호를 요구하지 않는 장. "후보 풀과 선정 과정"은 코드가 만든 발굴·관문 집계 표라 출처 인용이 없다
NO_CITE_SECTIONS = ("SUMMARY", "REFERENCE", "한계점", "후보 풀과 선정 과정")


class Judgment(BaseModel):
    groundedness: bool = Field(description="본문 주장마다 인용 번호가 있고 REFERENCE 의 출처로 뒷받침되는가")
    groundedness_reason: str = Field(description="판단 이유 한두 문장")
    neutrality: bool = Field(description="근거 없는 우열 판정·과장·평가 형용사 없이, 결론이 평가표 결과로만 제시되는가")
    neutrality_reason: str = Field(description="판단 이유 한두 문장")
    bias_control: bool = Field(description="단일 출처나 회사 자체 발표에 기대지 않고, 불리한 근거·미확인 사항도 함께 다루는가")
    bias_control_reason: str = Field(description="판단 이유 한두 문장")
    feedback: str = Field(description="미달 항목이 있으면 보고서 작성자가 고칠 점을 구체적으로. 모두 충족이면 빈 문자열")


def _source_key(s: dict) -> str:
    """출처 단위: 웹은 호스트, 문서는 발행 기관."""
    if s.get("kind") == "doc":
        return s.get("publisher") or s.get("doc_id") or "doc"
    return urlparse(s.get("url") or "").netloc.removeprefix("www.") or s.get("site") or "web"


def _own_site(s: dict, company: list[str]) -> bool:
    """회사 자체 사이트 근거인지 (호스트에 회사 이름이 들어 있음)."""
    host = norm(urlparse(s.get("url") or "").netloc)
    return s.get("kind") == "web" and any(len(k) >= 3 and k in host for k in company)


def rule_checks(report: dict, md: str, registry: dict, company: list[str] = ()) -> dict[str, list[str]]:
    """1안: 형식으로 확인할 수 있는 부분만 검사한다. company 는 보고서 대상 회사 이름(정규화). 반환 {항목: 실패 사유 목록}."""
    ev = get_config().evaluation
    checks = report.get("checks") or {}
    fails: dict[str, list[str]] = {k: [] for k in ITEMS}

    if not checks.get("refs_match_citations", False):
        fails["groundedness"].append("본문 인용 번호와 REFERENCE 목록이 일치하지 않음")
    if not checks.get("reference_format_ok", True):
        fails["groundedness"].append("REFERENCE 표기 형식 오류")
    sections = re.split(r"^## ", md, flags=re.M)[1:]
    bare = [sec.splitlines()[0] for sec in sections
            if not any(t in sec.splitlines()[0] for t in NO_CITE_SECTIONS) and not NUM_CITE.search(sec)]
    if bare:
        fails["groundedness"].append(f"인용이 없는 장: {', '.join(bare)}")
    if checks.get("chapter_refs_ok") is False:
        fails["groundedness"].append("SUMMARY 의 장 참조(→n장)가 없는 장을 가리킴")
    if checks.get("conclusion_matches_scorecard") is False:
        fails["groundedness"].append("결론 문장이 평가표 결과와 다름")
    if checks.get("assumptions_labeled") is False:
        fails["groundedness"].append("ROI 가정 수치에 '가정' 표기가 없음")

    if checks.get("consistency_violations"):
        fails["neutrality"].append(f"평가표와 어긋난 단정 {len(checks['consistency_violations'])}건: "
                                   f"{checks['consistency_violations'][0]}")
    if checks.get("banned_hits"):
        fails["neutrality"].append(f"SUMMARY 금지 표현: {', '.join(checks['banned_hits'])}")
    if checks.get("summary_rule_violations"):
        fails["neutrality"].append(f"SUMMARY 규칙 위반: {'; '.join(checks['summary_rule_violations'])}")

    cited = [s for i in report.get("cited_ids") or [] if (s := registry.get(i))]
    keys = [_source_key(s) for s in cited]
    distinct = set(keys)
    if len(distinct) < ev.min_distinct_sources:
        fails["bias_control"].append(f"서로 다른 출처 {len(distinct)}곳 < {ev.min_distinct_sources}곳")
    if keys:
        top = max(distinct, key=keys.count)
        share = keys.count(top) / len(keys)
        if share > ev.max_single_source_share:
            fails["bias_control"].append(f"단일 출처 편중: {top} {share:.0%} > {ev.max_single_source_share:.0%}")
    if cited:
        third = sum(not _own_site(s, company) for s in cited) / len(cited)
        if third < ev.min_third_party_share:
            fails["bias_control"].append(f"제3자 출처 비율 {third:.0%} < {ev.min_third_party_share:.0%} (회사 자체 사이트 편중)")
    return fails


def judge(md: str) -> Judgment:
    """2안: LLM Judge 가 내용을 판정한다."""
    cfg = get_config()
    return bounded(Judgment, "judge").invoke(
        render("report_eval", domain=cfg.domain.name, report=md[:cfg.evaluation.judge_max_chars]))


def evaluate_node(state: dict) -> dict:
    report = state.get("report") or {}
    revisions = state.get("report_revisions", 0)
    if report.get("mode") not in ("invest", "hold") or not report.get("md"):
        # 보정 실행(보고서 없음)과 적격 후보 없음(C안, LLM 없이 코드로 작성)은 평가할 본문이 없다
        result = {"passed": True, **{k: True for k in ITEMS}, "retry": False, "rule_failures": [], "judge_reasons": {},
                  "feedback": f"평가 대상 아님 (보고서 모드 {report.get('mode')})"}
        return {"report_evaluation": result, "log": [f"[품질 평가] {result['feedback']}"]}

    md = Path(report["md"]).read_text(encoding="utf-8")
    target = next((e for e in state.get("evaluations") or [] if e.get("name") == report.get("target")), {})
    company = company_keys({**(target.get("profile") or {}), "name": target.get("name")}) if target else []
    fails = rule_checks(report, md, state.get("registry") or {}, company)
    j = judge(md)
    items = {k: not fails[k] and getattr(j, k) for k in ITEMS}
    passed = all(items.values())
    rule_failures = [f"{LABELS[k]}: {x}" for k in ITEMS for x in fails[k]]
    llm_failures = [f"{LABELS[k]}: {getattr(j, k + '_reason')}" for k in ITEMS if not getattr(j, k)]
    feedback = "\n".join(f"- {x}" for x in rule_failures + llm_failures + ([j.feedback] if j.feedback and not passed else []))
    retry = not passed and revisions < get_config().evaluation.max_revisions
    result = {"passed": passed, **items, "retry": retry, "rule_failures": rule_failures,
              "judge_reasons": {k: getattr(j, k + "_reason") for k in ITEMS}, "feedback": "" if passed else feedback}
    status = " · ".join(f"{LABELS[k]} {'O' if v else 'X'}" for k, v in items.items())
    msg = (f"[품질 평가] {status} → "
           + ("통과" if passed else f"미달 → 보고서 재작성 ({revisions + 1}회째)" if retry
              else f"미달, 재작성 상한 {revisions}회 소진 → 평가 결과를 남기고 종료"))
    print(msg)
    return {"report_evaluation": result, "report_revisions": revisions + int(retry), "log": [msg]}


def route_after_evaluate(state: dict) -> str:
    """통과했거나 재작성 상한에 닿았으면 끝, 아니면 report 로 돌아가 다시 쓴다."""
    return "report" if (state.get("report_evaluation") or {}).get("retry") else "end"
