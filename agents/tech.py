"""기술 요약 에이전트 (Scorecard 제품/기술력 15%, 문항 P1~P4).

- 제품·핵심 기술·특허·실증·판매 근거를 웹에서 모으고, 홈페이지(못 읽으면 회사 기사)를 요약한다.
- 분야 기술 기준선은 문서 코퍼스에서 Agentic RAG 로 가져온다.
- 결과: 장점·단점·차별점 주장(claims, 경쟁사 에이전트가 검증)과 제품/기술력 판정.
evidence_blocks·CITE_ID·candidate_line·judge_own 은 다른 분석 에이전트도 쓴다.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from core import judge
from core.config import get_config, get_segment, run_date
from core.llm_bounded import bounded
from core.prompts import render
from rag import agentic_rag as rag
from rag.index import get_chunks
from tools import agent_tools, channels
from tools.fetch import enrich
from tools.grounding import norm
from tools.listing_check import normalize
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "tech"

# 기사 본문에서 기술·제품·실적 문단을 고르는 표현
TECH_TERMS = re.compile(r"기술|특허|제품|개발|실증|상용|출시|설치|도입|농가|매출|고객|계약|협약|공급|수출|정확도|절감|수확량|"
                        r"AI|인공지능|로봇|센서|patent|product|deploy|customer|revenue|trial|yield", re.I)
CITE_ID = re.compile(r"\b[WD][0-9a-f]{5,8}\b")
BASELINE_Q = "{seg} 분야의 기술 동향, 상용화 수준, 기술적 과제"
BASELINE_PURPOSE = "스타트업 기술 수준을 비교할 기준선"
HOMEPAGE_FOCUS = "핵심 기술·제품·장점과 단점"


def tech_queries(c: dict) -> list[tuple[str, bool]]:
    """기술·제품 검색 쿼리 (쿼리, deep)."""
    kr = c["region"] == "KR"
    name = c["official_name"] if kr else (c.get("name_en") or c["official_name"])
    return [(t["query"].format(name=name), t["deep"]) for t in get_config().domain.tech_queries["ko" if kr else "en"]]


def _covered(passage: str, skip: str) -> bool:
    """문단의 절반 이상이 이미 스니펫에 있으면 True (같은 내용을 두 번 넘기지 않게)."""
    parts = [passage[i:i + 30] for i in range(0, max(1, len(passage) - 30), 30)]
    return bool(skip) and sum(x in skip for x in parts) > len(parts) / 2


def body_passages(body: str, keys: list[str], terms: re.Pattern, n: int = 2, window: int = 400,
                  skip: str = "", boost: re.Pattern | None = None, person: re.Pattern | None = None) -> list[str]:
    """본문에서 회사명(keys)·용어(terms) 주변 ±window 자 문단을 최대 n 개 고른다 (서로 겹치지 않게).
    용어·boost 표현·사람 이름+직함(person, 창업자 에이전트가 넘김)이 많이 모이고 회사명이 함께 있는 문단을 먼저 고르고,
    스니펫(skip)에 이미 있는 문단은 건너뛴다."""
    low = body.lower()
    hits = {m.start() for m in terms.finditer(body)}
    for k in keys:
        i = low.find(k.lower())
        while i >= 0:
            hits.add(i)
            i = low.find(k.lower(), i + 1)

    def win(p: int) -> str:
        return body[max(0, p - window): p + window]

    def score(p: int) -> int:
        w = win(p)
        return (len(terms.findall(w)) + (2 * len(person.findall(w)) if person else 0)
                + (len(boost.findall(w)) if boost else 0) + 2 * any(k.lower() in w.lower() for k in keys))

    skip = re.sub(r"\s+", " ", skip or "")
    chosen: list[int] = []
    for p in sorted(hits, key=lambda p: (-score(p), p)):
        if len(chosen) >= n:
            break
        if any(abs(p - q) < 2 * window for q in chosen) or _covered(win(p), skip):
            continue
        chosen.append(p)
    return [("…" if p > window else "") + win(p) + ("…" if p + window < len(body) else "") for p in sorted(chosen)]


def evidence_blocks(reg: SourceRegistry, ids: list[str], keys: list[str], terms: re.Pattern = TECH_TERMS,
                    max_chars: int = 650, n: int = 2, window: int = 400, boost: re.Pattern | None = None,
                    person: re.Pattern | None = None) -> dict[str, str]:
    """근거 id → LLM 에 넘길 텍스트 (분석 에이전트 공용).
    - 웹 근거: 스니펫(max_chars) + 회사명(keys)이 나오는 근거는 본문 핵심 문단 최대 n 개(±window 자)
    - 문서 조각: 자르지 않고 그대로 (조각 800자 기준, 끝부분 수치가 잘리지 않게)"""
    keys = [k for k in keys if k and len(norm(k)) >= 2]
    out: dict[str, str] = {}
    for sid in dict.fromkeys(ids):
        s = reg.get(sid)
        if not s:
            continue
        if s["kind"] == "doc":
            out[sid] = reg.brief([sid], 1200)
            continue
        block = reg.brief([sid], max_chars)
        body = s.get("body") or ""
        if body and keys and any(norm(k) in norm(reg.text(sid)) for k in keys):
            ps = body_passages(body, keys, terms, n, window, skip=s["snippet"][:max_chars], boost=boost, person=person)
            if ps:
                block += "\n" + "\n".join(f"(본문) {p}" for p in ps)
        out[sid] = block
    return out


def cited(items: list[str], valid: set[str]) -> list[str]:
    """근거 저장소에 있는 근거 id 가 하나 이상 달린 항목만 남긴다."""
    return [x for x in items if set(CITE_ID.findall(x)) & valid]


def candidate_line(c: dict) -> str:
    """판정 LLM 에 넘기는 요약문 첫 줄: 후보 기본 정보."""
    return (f"[후보] {c['official_name']} | 단계 {c.get('stage')} ({c.get('round_date') or '시점 미상'}, "
            f"{c.get('round_amount') or '금액 미상'}) | 설립 {c.get('founded_year') or '확인 불가'} | {c.get('one_line')}")


def judge_own(dim: str, state: dict, reg: SourceRegistry, pool_ids: list[str], analysis: str) -> dict:
    """에이전트가 맡은 Scorecard 기준(dim)을 core.judge 로 판정한다.
    근거 풀 = 지금까지 모인 근거(current·앞 에이전트의 pool_ids) + 이 에이전트가 모은 근거.
    이 노드에서 새로 등록한 근거도 풀에 들어가도록 state 의 registry 대신 지금의 reg 로 거른다."""
    pool = judge.evidence_pool({**state, "registry": reg.data}, pool_ids)
    return judge.judge_dimension(dim, state["current"], pool, reg, analysis, state.get("run_date") or run_date())


def homepage_url(c: dict, reg: SourceRegistry, ids: list[str]) -> str | None:
    """회사 홈페이지 주소: TIPS 목록 → 회사 이름이 호스트에 들어간 웹 근거 URL → None."""
    if c.get("region") == "KR":
        names = {normalize(x) for x in (c.get("name"), c.get("official_name")) if x}
        for r in channels.tips_agtech(min_year=0):
            if normalize(r["name"]) in names and r.get("homepage"):
                url = r["homepage"].strip()
                return url if url.startswith("http") else "https://" + url
    keys = [k for k in (norm(c.get("name_en") or ""), norm(c.get("official_name") or "")) if len(k) >= 3]
    for sid in ids:
        s = reg.get(sid) or {}
        host = s.get("url", "").split("/")[2] if s.get("url", "").count("/") >= 2 else ""
        if s.get("kind") == "web" and any(k in norm(host) for k in keys):
            return s["url"]
    return None


def _main_article(reg: SourceRegistry, ids: list[str], keys: list[str]) -> str | None:
    """홈페이지를 못 읽었을 때 대신 요약할 회사 기사: 본문이 있고 회사명이 가장 많이 나오는 웹 근거."""
    best, top = None, 0
    for sid in ids:
        s = reg.get(sid) or {}
        body = norm(s.get("body") or "")
        n = sum(body.count(k) for k in keys) if s.get("kind") == "web" and len(body) >= 300 else 0
        if n > top:
            best, top = sid, n
    return best


def summarize_company(c: dict, reg: SourceRegistry, ids: list[str], keys: list[str]) -> tuple[str | None, str | None, str]:
    """홈페이지를 문서 요약 도구로 요약한다. 실패하면 회사 기사 본문 1건을 요약한다.
    반환 (요약문 또는 None, 요약한 출처(URL 또는 근거 id) 또는 None, 로그용 메모). 실패해도 예외를 던지지 않는다."""
    try:
        summarize = agent_tools.make_tools(reg, AGENT)["summarize_document"]
        url = homepage_url(c, reg, ids)
        if url and (text := summarize.invoke({"source": url, "focus": HOMEPAGE_FOCUS})):
            return text, url, "홈페이지 요약"
        sid = _main_article(reg, ids, keys)
        if sid and (text := summarize.invoke({"source": sid, "focus": HOMEPAGE_FOCUS})):
            return text, sid, "홈페이지 " + ("읽기 실패" if url else "주소 없음") + " → 회사 기사 요약"
        return None, None, "요약 없음"
    except Exception as e:  # 요약 실패로 분석을 멈추지 않는다
        return None, None, f"요약 실패({type(e).__name__})"


class TechAnalysis(BaseModel):
    product: str = Field(description="주요 제품·서비스 (근거 id 인용 [W..])")
    core_technology: str = Field(description="핵심 기술과 AI 가 하는 일 (근거 id)")
    maturity: Literal["연구", "시제품", "실증", "상용", "확인 불가"]
    maturity_evidence: str = Field(description="성숙도 근거와 그 날짜 (근거 id), 앞으로의 계획은 '예정'으로 구분")
    pros: list[str] = Field(description="기술·제품의 장점 2~4개, 각 항목 끝에 근거 id. 회사 발표뿐이면 '회사 측 주장:' 으로 시작")
    cons: list[str] = Field(description="기술·제품의 단점·한계·위험 1~4개, 각 항목 끝에 근거 id")
    claims: list[str] = Field(
        description="회사가 내세우는 경쟁 차별점 주장 1~4개 (경쟁사 비교 에이전트가 검증). 형식 '<주장> [근거 id]'")
    ip_evidence: str = Field(description="특허·논문·인증 근거 (등록·출원 구분, 근거 id), 없으면 '확인 불가'")
    license_check: str = Field(
        description="제품 유형상 필요한 인허가·검정(농업기계 검정, 농약·비료·동물용 의약품 등록 등)과 취득·신청 여부 (근거 id). "
                    "대상이 아니면 '해당 없음: <사유>', 근거가 없으면 '확인 불가'")
    growth_signals: list[str] = Field(
        description="제품·사업 성장 신호 0~5개: 상용 설치·유료 고객·계약·수출·출시. 형식 '<YYYY-MM> <사건> [근거 id]'")
    trend_fit: str = Field(description="분야 기술 기준선(문서 근거 [D..]) 대비 이 회사 기술의 위치")
    evidence_ids: list[str] = Field(description="실제로 인용한 근거 id 전체")


def _analysis_text(c: dict, t: dict) -> str:
    """제품/기술력 판정에 넘기는 기술 요약문."""
    return "\n".join([
        candidate_line(c),
        f"[기술] 제품: {t['product']} / 핵심 기술: {t['core_technology']} / 성숙도: {t['maturity']} "
        f"({t['maturity_evidence']}) / 특허·인증: {t['ip_evidence']} / 인허가: {t['license_check']}",
        f"[장점] {'; '.join(t['pros']) or '확인 불가'}",
        f"[단점] {'; '.join(t['cons']) or '확인 불가'}",
        f"[회사 측 차별점 주장] {'; '.join(t['claims']) or '없음'}",
        f"[성장 신호] {'; '.join(t['growth_signals']) or '없음'}",
        f"[업계 기준선 대비] {t['trend_fit']}",
    ])


def tech_node(state: dict) -> dict:
    c = state["current"]
    reg = SourceRegistry(state.get("registry"))
    name = c["official_name"]
    seg = get_segment(c["segment_id"])
    ids = list(c.get("evidence_ids", []))
    # 특허·실증·판매·파트너 검색
    for query, deep in tech_queries(c):
        ids += web_search(query, reg, AGENT, topic="news", recent=False, deep=deep, raw=True)
    # 기사 원문으로 스니펫에 없는 사실 보강
    enrich(reg, ids, [norm(name), norm(c.get("name_en") or "")], limit=12)
    # 코퍼스에서 회사 이름이 나오는 조각 (제3자 근거)
    keys = [k for k in (norm(name), norm(c.get("name_en") or "")) if len(k) >= 2]
    for ch in get_chunks():
        if any(k in norm(ch.page_content) for k in keys):
            ids.append(reg.add_doc(ch.metadata, ch.page_content, agent=AGENT, query=f"코퍼스 속 {name}"))
    # 분야 기술 기준선 (Agentic RAG)
    base = rag.answer_question(BASELINE_Q.format(seg=seg["name"]), BASELINE_PURPOSE, reg, AGENT, allow_direct=False)
    ids += base["evidence_ids"]
    # 홈페이지 요약
    summary, summary_src, summary_note = summarize_company(c, reg, ids, keys)
    if summary:
        ids += [i for i in CITE_ID.findall(summary) if reg.get(i)]
    ids = list(dict.fromkeys(ids))

    blocks = evidence_blocks(reg, ids, [name, c.get("name_en") or ""])
    ctx = dict(name=name, one_line=c.get("one_line", ""), segment=seg["name"],
               baseline=base.get("answer") or "(생성 답변 없음 — 아래 문서 근거 [D...] 를 직접 읽을 것)",
               homepage=summary or "(요약 없음)", evidence="\n\n".join(blocks.values()))
    res: TechAnalysis = bounded(TechAnalysis).invoke(render("tech", **ctx))

    valid = set(ids)
    out = res.model_dump()
    # 근거 id 가 달린 항목만 남긴다
    out["growth_signals"] = cited(res.growth_signals, valid)
    out["claims"] = cited(res.claims, valid)
    refs = (list(res.evidence_ids) + [i for x in out["growth_signals"] + out["claims"] + res.pros + res.cons
                                      for i in CITE_ID.findall(x)])
    out["evidence_ids"] = [i for i in dict.fromkeys(refs) if i in valid]
    out["baseline_answer"] = base.get("answer") or ""
    out["homepage_summary"] = summary
    out["homepage_source"] = summary_src
    out["pool_ids"] = ids
    out["criterion"] = judge_own("product", state, reg, ids, _analysis_text(c, out))
    crit = out["criterion"]
    msg = (f"[기술 요약] {name}: 성숙도 {res.maturity}, 장점 {len(res.pros)}·단점 {len(res.cons)}·주장 {len(out['claims'])}건, "
           f"{summary_note}, 기준선 {base.get('status')}, 근거 {len(out['evidence_ids'])}건 → 제품/기술력 "
           f"YES {crit['yes']}/{crit['n']}")
    print(msg)
    return {"registry": reg.data, "tech": out, "log": [msg],
            "rag_traces": [{"agent": AGENT, "question": "기술 동향", "route": base.get("route"),
                            "status": base.get("status"), "trace": base.get("trace", [])}]}
