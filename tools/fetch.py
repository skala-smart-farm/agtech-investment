"""기사 원문 수집. 검색 스니펫에 없는 본문과 원문 게시일을 근거에 채운다 (결과는 캐시에 저장)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from core.config import get_config, path
from tools.sources import SourceRegistry

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0 Safari/537.36"}
SKIP = ("thevc.kr", "innoforest.co.kr", "linkedin.com", "jointips.or.kr", "data.go.kr", "youtube.com")


def _cache(url: str):
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return path(f"{get_config().cache.dir}/articles/{h}.json")


def cached(url: str) -> dict | None:
    """캐시에 받아 둔 원문만 읽는다. 없거나 실패 기록이면 None."""
    if not (url or "").startswith("http"):
        return None
    f = _cache(url)
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return None if d.get("error") else d


def fetch(url: str) -> dict | None:
    """원문 본문·게시일·제목. 실패하면 None (실패도 캐시)."""
    host = url.split("/")[2].lower() if url.count("/") >= 2 else ""
    if not url.startswith("http") or url.lower().endswith(".pdf") or host.endswith(SKIP):
        return None
    f = _cache(url)
    if f.exists():
        return cached(url)
    if os.getenv("REPLAY_OFFLINE"):
        return None
    import requests
    import trafilatura

    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        meta = trafilatura.extract(r.text, url=url, with_metadata=True, output_format="json")
        d = json.loads(meta) if meta else {}
        out = {"text": re.sub(r"\s+", " ", d.get("text") or "")[:6000], "date": d.get("date") or "",
               "title": (d.get("title") or "").strip()}
        if len(out["text"]) < 200:
            raise ValueError("본문이 너무 짧음")
    except Exception as e:
        f.write_text(json.dumps({"error": str(e)[:200]}, ensure_ascii=False), encoding="utf-8")
        return None
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def enrich(reg: SourceRegistry, ids: list[str], keys: list[str], limit: int = 10) -> int:
    """회사명이 제목·스니펫에 나오는 웹 근거 중 본문이 없는 것을 골라 원문을 채운다. 채운 개수를 돌려준다."""
    from tools.grounding import norm

    targets = []
    for i in ids:
        s = reg.get(i)
        if not s or s["kind"] != "web" or s.get("body"):
            continue
        if any(k and k in norm(s["title"] + s["snippet"]) for k in keys):
            targets.append(i)
    targets = targets[:limit]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda i: fetch(reg.get(i)["url"]), targets))
    n = 0
    for i, d in zip(targets, results):
        if not d:
            continue
        s = reg.data[i]
        s["body"] = d["text"]
        if not s.get("date") and re.match(r"20\d{2}-\d{2}-\d{2}", d["date"]):
            s["date"], s["date_is_access"] = d["date"][:10], False
        n += 1
    return n
