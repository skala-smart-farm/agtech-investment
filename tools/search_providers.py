"""검색 공급자. 키가 있는 공급자만 Serper → Tavily 순서로 시도한다.

결과는 같은 형식 {url, title, content, published_date} 으로 맞추고, 오류는 ProviderError 로 올린다.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta

import requests

HANGUL = re.compile(r"[가-힣]")
TAVILY_URL = "https://api.tavily.com/search"
# 인증 실패·한도 초과 → 이번 실행에서 그 공급자를 끈다
FATAL_STATUS = {401, 403, 432, 433}


class ProviderError(RuntimeError):
    """검색 공급자 오류. status 는 HTTP 상태 코드(연결·시간 초과는 None)."""

    def __init__(self, provider: str, message: str, status: int | None = None, timed_out: bool = False,
                 retry_after: float = 0.0):
        super().__init__(message)
        self.provider, self.status, self.timed_out, self.retry_after = provider, status, timed_out, retry_after

    @property
    def fatal(self) -> bool:
        return self.status in FATAL_STATUS

    @property
    def retryable(self) -> bool:
        """다시 시도할 만한 오류: 연결 실패, 429, 5xx."""
        return not self.timed_out and (self.status is None or self.status == 429 or self.status >= 500)


def order_for(query: str) -> list[str]:
    return [p for p, key in (("serper", "SERPER_API_KEY"), ("tavily", "TAVILY_API_KEY")) if os.getenv(key)]


def _serper_date(raw: str | None, today: str) -> str | None:
    """Serper 날짜 표기("3일 전", "Jan 5, 2025" 등)를 YYYY-MM-DD 로 바꾼다."""
    if not raw:
        return None
    base = date.fromisoformat(today)
    m = re.search(r"(\d+)\s*(minute|hour|day|week|month|year|분|시간|일|주|개월|달|년)", raw)
    if m and ("ago" in raw or "전" in raw):
        n, unit = int(m.group(1)), m.group(2)
        days = {"minute": 0, "분": 0, "hour": 0, "시간": 0, "day": 1, "일": 1, "week": 7, "주": 7,
                "month": 30, "개월": 30, "달": 30, "year": 365, "년": 365}[unit]
        return (base - timedelta(days=n * days)).isoformat()
    m = re.search(r"(20\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def tavily(query: str, topic: str, time_range: str | None, max_results: int, include_domains: list[str] | None,
           exclude_domains: list[str], depth: str, raw: bool, timeout: float) -> list[dict]:
    """Tavily 검색 REST 호출 (timeout 적용). 결과가 없으면 []."""
    body = {"query": query, "max_results": max_results, "search_depth": depth, "include_domains": include_domains or [],
            "exclude_domains": [] if include_domains else list(exclude_domains), "include_images": False, "topic": topic}
    if raw:
        body["include_raw_content"] = "text"  # 본문 전체
    if time_range:
        body["time_range"] = time_range
    try:
        r = requests.post(TAVILY_URL, json=body, timeout=(min(10, timeout), timeout),
                          headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"})
    except requests.RequestException as e:
        raise ProviderError("tavily", type(e).__name__, None, isinstance(e, requests.Timeout)) from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", {})
            msg = detail.get("error") if isinstance(detail, dict) else str(detail)
        except ValueError:
            msg = r.text[:80]
        wait = r.headers.get("Retry-After", "") if r.status_code == 429 else ""
        raise ProviderError("tavily", f"Error {r.status_code}: {msg}", r.status_code,
                            retry_after=float(wait) if wait.replace(".", "", 1).isdigit() else 0.0)
    return r.json().get("results", []) or []


def serper(query: str, topic: str, include_domains: list[str] | None,
           time_range: str | None, today: str) -> list[dict]:
    q = query + (" (" + " OR ".join(f"site:{d}" for d in include_domains) + ")" if include_domains else "")
    body = {"q": q, "num": 10}
    if HANGUL.search(query):
        body.update(gl="kr", hl="ko")
    if time_range == "year":
        body["tbs"] = "qdr:y"
    endpoint = "news" if topic == "news" else "search"
    try:
        r = requests.post(f"https://google.serper.dev/{endpoint}", json=body,
                          headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=20)
    except requests.RequestException as e:
        raise ProviderError("serper", type(e).__name__, None, isinstance(e, requests.Timeout)) from e
    if not r.ok:
        raise ProviderError("serper", f"Error {r.status_code}: {r.text[:80]}", r.status_code)
    items = r.json().get("news" if endpoint == "news" else "organic", [])
    return [{"url": it["link"], "title": it.get("title", ""), "content": it.get("snippet", ""),
             "published_date": _serper_date(it.get("date"), today)} for it in items if it.get("link")]
