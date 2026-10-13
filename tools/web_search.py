"""웹·뉴스 검색 (Serper → Tavily, 키가 있는 공급자만).

- 뉴스는 최근 1년만 검색하고, 결과가 부족할 때만 기간 제한을 푼다.
- 같은 쿼리는 디스크 캐시를 재사용한다.
- 공급자가 인증·한도 오류이거나 연속으로 실패하면 이번 실행에서 끄고 다음 공급자로 넘어간다.
- 실패한 검색은 "결과 없음"과 구분해 FAILED_QUERIES 에 남긴다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import date

from core.config import get_config, path, require_keys
from tools import search_providers as providers
from tools.sources import SourceRegistry, today

# 유료 시장조사 홍보 기사
PR_MARKET = re.compile(r"market (size|share|growing|to reach|worth|report|intelligence|analysis|forecast)|cagr|"
                       r"says .*(research|intelligence|insights)|(mordor|marketsandmarkets|grand view|precedence|imarc|"
                       r"fortune business|researchandmarkets|research and markets)", re.I)
# 검색 결과 목록 페이지
SEARCH_PAGE = re.compile(r"//search\.|/search[/?]|[?&](kwd|q|query|keyword)=", re.I)
# 실패한 검색 {query, agent}. 다른 모듈이 같은 객체를 참조하므로 다시 대입하지 않는다
FAILED_QUERIES: list[dict] = []
# 끈 공급자·연속 오류 수·검색 횟수 (병렬 호출이라 잠금)
DISABLED: dict[str, str] = {}
_STREAK: dict[str, int] = {}
_LIVE = {"n": 0, "sec": 0.0}
_LOCK = threading.Lock()


def _cache_file(key: dict):
    h = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return path(f"{get_config().cache.dir}/search/{h}.json")


def _legacy_files(key: dict) -> list:
    """예전 캐시 키 형식의 파일 후보."""
    ex = list(get_config().search.exclude_domains)
    return [_cache_file({**key, "ex": ex[:n]}) for n in (15, 33, len(ex))]


def _raw_search(query: str, topic: str, time_range: str | None, max_results: int,
                include_domains: list[str] | None, depth: str = "basic", raw: bool = False) -> list[dict] | None:
    """검색 결과 목록. 검색이 실패하면 None, 결과가 없으면 []."""
    cfg = get_config()
    key = {"q": query, "topic": topic, "tr": time_range, "n": max_results, "dom": include_domains or [],
           "depth": depth, "raw": raw}
    f = _cache_file(key)
    if cfg.cache.search and not f.exists():
        old = next((o for o in _legacy_files(key) if o.exists()), None)
        if old:
            f.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    if cfg.cache.search and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    failed = f.with_suffix(".failed")
    # 캐시에 실패로 남은 검색은 --retry-failed 일 때만 다시 시도
    if cfg.cache.search and failed.exists() and not os.getenv("SEARCH_RETRY_FAILED"):
        return None
    if os.getenv("REPLAY_OFFLINE"):
        raise RuntimeError(f"--offline: 재현용 캐시에 없는 검색입니다 → {query!r}")
    require_keys()
    results, ok, errors, t0 = [], False, [], time.time()
    active = active_providers()
    if not active:
        errors.append(f"공급자 모두 꺼짐({', '.join(DISABLED)})")
    for provider in active:
        try:
            got = _call_with_retry(provider, provider == active[-1], query, topic, time_range, max_results,
                                   include_domains, depth, raw)
        except Exception as e:  # 한도 초과·인증 오류 → 다음 공급자
            errors.append(f"{provider}: {str(e)[:60]}")
            _note_error(provider, e)
            continue
        ok = True
        results = [{**r, "provider": provider} for r in got]
        if results:
            break
    if not ok:
        print(f"   (검색 실패, 빈 결과로 진행: {query[:40]} — {'; '.join(errors)[:120]})")
    _progress(time.time() - t0)
    if cfg.cache.search and ok:
        f.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        failed.unlink(missing_ok=True)
    elif cfg.cache.search:  # 오류는 실패 표시만 남긴다
        failed.write_text(json.dumps({"query": query, "error": "search failed"}, ensure_ascii=False), encoding="utf-8")
    return results if ok else None


def active_providers() -> list[str]:
    """키가 있고 이번 실행에서 꺼지지 않은 공급자 (Serper → Tavily 순서)."""
    return [p for p in providers.order_for("") if p not in DISABLED]


def _note_error(provider: str, e: Exception) -> None:
    """인증·한도 오류이거나 연속 오류가 search.breaker_after 번이면 그 공급자를 끈다."""
    fatal = getattr(e, "fatal", False)
    if getattr(e, "status", None) == 429:  # 429 는 세지 않는다
        return
    with _LOCK:
        _STREAK[provider] = _STREAK.get(provider, 0) + 1
        if provider in DISABLED or not (fatal or _STREAK[provider] >= int(get_config().search.get("breaker_after", 3))):
            return
        DISABLED[provider] = str(e)[:80]
        streak = _STREAK[provider]
    why = "인증·사용 한도 오류" if fatal else f"연속 {streak}번 오류"
    print(f"   [검색] {provider} 를 이번 실행에서 끕니다 ({why}: {str(e)[:60]}). 남은 공급자가 없으면 이후 검색은 실패로 기록되고 "
          f"해당 판정은 fail-closed 로 처리됩니다 → 키·한도를 확인한 뒤 --retry-failed 로 실패한 검색만 다시 할 수 있습니다")


def _call_with_retry(provider: str, last: bool, *args) -> list[dict]:
    """일시 오류(연결 실패·429·5xx)만 search.retries 번 다시 시도한다."""
    retries = int(get_config().search.get("retries", 1)) if last else 0
    for attempt in range(retries + 1):
        try:
            got = _call(provider, *args)
        except providers.ProviderError as e:
            if not e.retryable or attempt == retries:
                raise
            time.sleep(min(10.0, e.retry_after) or 2 * (attempt + 1))
            continue
        with _LOCK:
            _STREAK[provider] = 0
        return got
    return []


def _progress(sec: float) -> None:
    """실시간 검색 search.progress_every 회마다 진행 상황 출력."""
    every = int(get_config().search.get("progress_every", 10) or 0)
    with _LOCK:
        _LIVE["n"] += 1
        _LIVE["sec"] += sec
        n, avg = _LIVE["n"], _LIVE["sec"] / _LIVE["n"]
    if every and (n == 1 or n % every == 0):
        print(f"   (실시간 웹 검색 {n}회 · 공급자 {' → '.join(active_providers()) or '없음'} · 평균 {avg:.1f}초/회)")


def _call(provider: str, query: str, topic: str, time_range: str | None, max_results: int,
          include_domains: list[str] | None, depth: str, raw: bool) -> list[dict]:
    cfg = get_config()
    if provider == "tavily":
        return providers.tavily(query, topic, time_range, max_results, include_domains, cfg.search.exclude_domains,
                                depth, raw, float(cfg.search.get("timeout_sec", 30)))
    blocked = tuple(cfg.search.exclude_domains)
    got = [r for r in providers.serper(query, topic, include_domains, time_range, date.today().isoformat())
           if not _host(r["url"]).endswith(blocked) and not PR_MARKET.search(r["title"])
           and not SEARCH_PAGE.search(r["url"])][:max_results]
    if raw:  # 본문은 원문 페이지에서 직접 받는다
        from tools.fetch import fetch

        for r in got:
            if d := fetch(r["url"]):
                r["raw_content"] = d["text"]
                r["published_date"] = r["published_date"] or d["date"][:10] or None
    return got


def _host(url: str) -> str:
    return url.split("/")[2].lower() if url.count("/") >= 2 else ""


def web_search(query: str, registry: SourceRegistry, agent: str, *, topic: str = "news",
               recent: bool = True, max_results: int | None = None,
               include_domains: list[str] | None = None, min_results: int = 2, deep: bool = False,
               raw: bool = False) -> list[str]:
    """검색 결과를 근거로 등록하고 근거 id 목록을 돌려준다.
    deep=True: 특정 회사에 대한 검색처럼 본문 근거가 더 필요할 때 Tavily advanced 검색(관련 문단을 더 길게 가져옴).
    raw=True : 기사 본문 전체도 받아 근거 본문(body)으로 저장한다."""
    cfg = get_config()
    n = max_results or cfg.search.max_results
    tr = cfg.search.news_time_range if (recent and topic == "news") else None
    depth = "advanced" if deep else "basic"
    got = [_raw_search(query, topic, tr, n, include_domains, depth, raw)]
    if len(got[0] or []) < min_results and tr:
        got.append(_raw_search(query, topic, cfg.search.fallback_time_range, n, include_domains, depth, raw))
    results = [r for g in got for r in g or []]
    if not results and got[-1] is None:  # 검색 실패를 '결과 없음'과 구분해 기록
        FAILED_QUERIES.append({"query": query, "agent": agent})
    ids: list[str] = []
    blocked = tuple(cfg.search.exclude_domains)
    for r in results:
        host = _host(r.get("url", ""))
        if (not r.get("url") or host.endswith(blocked) or PR_MARKET.search(r.get("title") or "")
                or SEARCH_PAGE.search(r["url"])):
            continue
        sid = registry.add_web(r, agent=agent, query=query, access_date=today())
        if sid not in ids:
            ids.append(sid)
    return ids
