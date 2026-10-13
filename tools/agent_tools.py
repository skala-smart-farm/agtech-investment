"""에이전트가 쓰는 도구 3개 (LangChain @tool).

- search_documents(query): 문서 코퍼스(PDF) 하이브리드 검색
- web_search(query, recent): 웹·뉴스 검색, 결과를 근거 저장소에 등록
- summarize_document(source, focus): 근거 id 또는 URL 의 원문 요약 (LLM 에 바인딩하지 않고 코드에서 직접 부름)

도구의 docstring 은 LLM 에 그대로 전달되는 설명이다.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from core.config import get_config
from core.llm import get_llm
from core.prompts import render
from rag.index import get_hybrid_retriever
from rag.loader import corpus_summary
from tools.fetch import fetch
from tools.sources import SourceRegistry, today
from tools.web_search import web_search as search_web

TOOL_NAMES = ("search_documents", "web_search", "summarize_document")


def make_tools(reg: SourceRegistry, agent: str) -> dict[str, BaseTool]:
    """에이전트(agent 이름은 근거 등록 기록용)가 쓸 도구 3개를 {이름: 도구} 로 돌려준다. 키는 TOOL_NAMES 와 같다."""

    def search_documents(query: str) -> tuple[str, list[dict]]:
        """공공·연구기관 AgTech 문서 코퍼스 검색 (PDF {corpus}, 한국어·영어).
        시장 규모·성장률, 정책·법령·지원 제도, 기술 성숙도·기술 기준선, 애그테크 투자 동향에 강하다.
        개별 스타트업의 매출·팀·최근 소식은 거의 없다 → web_search.

        Args:
            query: 보고서에 쓰일 핵심 명사·지표로 쓴 검색어 (예: 스마트팜 시장 규모 연평균 성장률)
        """
        docs = get_hybrid_retriever().invoke(query)[: get_config().rag.candidate_k]
        chunks = [{"text": d.page_content, "meta": d.metadata} for d in docs]
        listing = "\n\n".join(
            f"[{i}] ({c['meta'].get('publisher')} {c['meta'].get('year')}, p.{c['meta'].get('page')})\n{c['text'][:500]}"
            for i, c in enumerate(chunks))
        return listing or "검색 결과 없음", chunks

    # 코퍼스 규모는 data/manifest.yaml 에서 계산해 넣는다
    search_documents.__doc__ = search_documents.__doc__.replace("{corpus}", corpus_summary())
    search_documents = tool(response_format="content_and_artifact", parse_docstring=True)(search_documents)

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def web_search(query: str, recent: bool = True) -> tuple[str, list[str]]:
        """웹·뉴스 검색. 개별 기업(스타트업·경쟁사)의 투자 유치·제품·계약·인물 소식과,
        문서 코퍼스 발행 이후의 최근 사건(정책 발표·보급 사례)을 찾는다.
        유료 시장조사 홍보 기사와 개인 블로그는 걸러지고, 결과는 [W…] 근거 id 로 등록된다.

        Args:
            query: 한국어 또는 영어 검색어 (단어 3~6개)
            recent: True 면 최근 1년 뉴스를 우선하고(결과가 부족하면 기간 제한을 푼다), False 면 기간 제한 없이 찾는다
        """
        ids = search_web(query, reg, agent, topic="news", recent=recent)
        return reg.brief(ids, 300) or "검색 결과 없음", ids

    @tool(parse_docstring=True)
    def summarize_document(source: str, focus: str) -> str:
        """문서 요약. 근거 id 또는 URL 의 원문을 focus 관점에서 5문장 이내로 요약하고 끝에 [근거 id] 를 붙인다.
        회사 주장과 제3자 사실을 구분한다. 원문을 읽지 못하면 빈 문자열을 돌려준다.

        Args:
            source: 근거 id(예: W1a2b3) 또는 http(s) URL
            focus: 요약 관점 (예: 핵심 기술과 장단점)
        """
        try:
            return summarize(reg, agent, source, focus)
        except Exception as e:  # 도구 실패가 에이전트를 멈추지 않게
            print(f"   (문서 요약 실패, 빈 결과로 진행: {source[:60]} — {str(e)[:80]})")
            return ""

    return {"search_documents": search_documents, "web_search": web_search, "summarize_document": summarize_document}


def summarize(reg: SourceRegistry, agent: str, source: str, focus: str) -> str:
    """summarize_document 본체. URL 이면 원문을 받아 근거로 등록하고, 근거 id 면 등록된 본문을 요약한다."""
    cfg = get_config().tools
    source = (source or "").strip()
    if source.startswith("http"):
        d = fetch(source)
        if not d or not d.get("text"):
            return ""
        sid = reg.add_web({"url": source, "title": d.get("title") or source, "content": d["text"][:1200],
                           "raw_content": d["text"], "published_date": d.get("date") or None},
                          agent=agent, query=f"문서 요약: {focus}", access_date=today())
        text = d["text"]
    elif source in reg.data:
        sid, text = source, reg.text(source)
    else:
        return ""
    text = text[: cfg.summarize_max_chars]
    step = cfg.summarize_chunk_chars
    parts = [p for p in (text[i: i + step] for i in range(0, len(text), step)) if p.strip()][:2]
    llm = get_llm("generator").bind(max_tokens=get_config().models.max_output_tokens["summarize"])  # 반복 퇴행 방지
    outs = [llm.invoke(render("summarize", focus=focus, text=p)).content.strip() for p in parts]
    summary = " ".join(o for o in outs if o)
    return f"{summary} [{sid}]" if summary else ""
