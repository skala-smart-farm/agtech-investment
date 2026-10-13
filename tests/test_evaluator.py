"""보고서 품질 평가 노드: 규칙 검사(1안)와 LLM Judge(2안)를 합친 판정."""
from __future__ import annotations

import agents.evaluator as E

MD = """# 보고서

## SUMMARY
- 결론: 투자 추천 [1]

## 1. 사업 아이디어·기술
상용 운영 중이며 실증을 마쳤다 [1, 2]

## 2. 시장 규모와 성장성
시장 규모는 성장 중이다 [3]

## 3. 팀의 구성
창업자와 경쟁사, 현장 도입 사례 [2]

## REFERENCE
1. 출처
"""
REG = {"W1": {"kind": "web", "url": "https://a.com/x"}, "W2": {"kind": "web", "url": "https://b.com/y"},
       "D1": {"kind": "doc", "publisher": "농촌진흥청"}}
GOOD = {"checks": {"refs_match_citations": True, "reference_format_ok": True, "chapter_refs_ok": True,
                   "conclusion_matches_scorecard": True, "assumptions_labeled": True, "consistency_violations": [],
                   "banned_hits": [], "summary_rule_violations": []}, "cited_ids": ["W1", "W2", "D1"]}


def test_rules_pass_on_well_formed_report():
    assert all(not v for v in E.rule_checks(GOOD, MD, REG).values())


def test_rules_catch_each_item():
    bad = {"checks": {"refs_match_citations": False, "consistency_violations": ["C2 가 YES 가 아닌데 '앞선다'"]},
           "cited_ids": ["W1", "W1", "W1", "W2"]}
    md = MD.replace("[3]", "")
    fails = E.rule_checks(bad, md, REG)
    assert fails["groundedness"] and any("인용이 없는 장" in x for x in fails["groundedness"])
    assert fails["neutrality"]
    assert any("단일 출처 편중" in x for x in fails["bias_control"])
    assert any("서로 다른 출처" in x for x in fails["bias_control"])


def test_hybrid_needs_both_rule_and_judge(tmp_path, monkeypatch, set_cfg):
    set_cfg("evaluation.max_revisions", 1)
    md = tmp_path / "r.md"
    md.write_text(MD, encoding="utf-8")
    report = {**GOOD, "mode": "invest", "md": str(md)}
    verdict = {k: True for k in E.ITEMS} | {f"{k}_reason": "이유" for k in E.ITEMS} | {"feedback": ""}
    monkeypatch.setattr(E, "judge", lambda _: E.Judgment(**(verdict | {"neutrality": False, "feedback": "우열 표현 삭제"})))
    out = E.evaluate_node({"report": report, "registry": REG})
    ev = out["report_evaluation"]
    assert not ev["passed"] and not ev["neutrality"] and ev["groundedness"]  # 규칙은 통과, LLM 이 중립성 미달
    assert ev["retry"] and out["report_revisions"] == 1 and "우열 표현 삭제" in ev["feedback"]
    assert E.route_after_evaluate(out) == "report"

    out = E.evaluate_node({"report": report, "registry": REG, "report_revisions": 1})
    assert not out["report_evaluation"]["retry"] and E.route_after_evaluate(out) == "end"  # 상한이면 끝낸다

    monkeypatch.setattr(E, "judge", lambda _: E.Judgment(**verdict))
    out = E.evaluate_node({"report": report, "registry": REG})
    assert out["report_evaluation"]["passed"] and E.route_after_evaluate(out) == "end"


def test_no_report_body_is_not_evaluated():
    out = E.evaluate_node({"report": {"mode": "calibrate"}})
    assert out["report_evaluation"]["passed"] and E.route_after_evaluate(out) == "end"


def test_strengthened_rules():
    checks = {**GOOD["checks"], "conclusion_matches_scorecard": False, "assumptions_labeled": False,
              "summary_rule_violations": ["SUMMARY 570자 > 560자"]}
    reg = {**REG, "W3": {"kind": "web", "url": "https://www.acmefarm.com/about"},
           "W4": {"kind": "web", "url": "https://acmefarm.com/news"}}
    fails = E.rule_checks({"checks": checks, "cited_ids": ["W3", "W4", "W1"]}, MD, reg, ["acmefarm"])
    assert any("결론 문장" in x for x in fails["groundedness"]) and any("가정" in x for x in fails["groundedness"])
    assert any("SUMMARY 규칙 위반" in x for x in fails["neutrality"])
    assert any("제3자 출처 비율 33%" in x for x in fails["bias_control"])
