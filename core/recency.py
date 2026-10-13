"""최근성 규칙. 판정·탐색 관문·발굴 채널이 함께 쓴다.

- 최근 N개월(config recency.months) = [평가 기준일 − N개월, 평가 기준일]. 기준일 이후 날짜(예정·계획)는 최근으로 인정하지 않는다.
- 날짜는 적힌 정밀도(연·월)로만 비교하고, 읽지 못하면 판단하지 않는다.
"""
from __future__ import annotations

import re

from core.config import get_config


def recent_months() -> int:
    """'최근'으로 인정하는 기간(개월)."""
    return int(get_config().recency.months)


def recent_days() -> int:
    return recent_months() * 365 // 12

# 라운드 시점·마일스톤 날짜 필드: 'YYYY', 'YYYY-MM', 'YYYY.MM', 'YYYY년 M월'
_FIELD = re.compile(r"((?:19|20)\d{2})(?:\s*[-./년]\s*(\d{1,2}))?")
# 문장 속 사건 날짜 (달 표시가 분명한 것만)
_TEXT_YM = re.compile(r"((?:19|20)\d{2})\s*(?:년\s*(\d{1,2})\s*월|[.\-/]\s*(\d{1,2})(?![\d,]))")
_TEXT_Y = re.compile(r"((?:19|20)\d{2})\s*년(?!\s*\d|형)")   # 연도만: '2027년까지'
_TEXT_RANGE = re.compile(r"((?:19|20)\d{2})\s*[~\-–]\s*(?:19|20)\d{2}\s*년")   # '2024~2027년' 의 시작 연도
# 영어 날짜: 'March 2027', 'by 2027'
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_EN_YM = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
                    r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?\s+(?:\d{1,2},?\s+)?((?:19|20)\d{2})\b", re.I)
_EN_Y = re.compile(r"\b(?:in|by|until|through|during|since|from|early|late|mid)[\s-]+((?:19|20)\d{2})\b(?![\d,.]\d)", re.I)

# 라운드 자체가 예정·추진인 문장
ROUND_PLANNED = re.compile(
    r"(?:투자\s*유치|유치|라운드|시리즈\s*[a-z]?|펀딩)\s*(?:를|을)?\s*(?:추진|예정|계획|목표|준비|앞두)"
    r"|(?:추진|예정|계획|목표|준비)\s*(?:중인|하는|한)\s*(?:투자|라운드|시리즈)"
    r"|\b(?:plans?|aims?|expects?|looks?|looking|seeks?|seeking|in talks)\s+to\s+(?:raise|close)\b"
    r"|\bis\s+(?:currently\s+)?raising\s+(?:(?:a|an|its|new|funds|capital|money|series|seed)\b|\$)"
    r"|\bwill\s+(?:raise|close)\b", re.I)
# 끝난 라운드 표현
ROUND_DONE = re.compile(r"유치했|유치하며|유치한\s|유치에\s*성공|유치\s*[…,.·!]|유치\s*$|받았|확보했|마무리|마쳤"
                        r"|\b(?:raised|closed|secured|completed|landed|bagged)\b", re.I)

# 예정·계획·목표 표현 (서술어로 쓰인 것만)
_RIEUL = "[" + "".join(chr(0xAC00 + i * 28 + 8) for i in range(19 * 21)) + "]"   # 받침 ㄹ 글자(할·될·늘릴…)
PLANNED = re.compile(
    rf"{_RIEUL}\s*(?:계획|예정|방침|전망)"                       # 설치할 계획, 출시될 예정, 늘릴 방침
    r"|(?:예정|계획)(?:이다|이며|이고|이라|입니다|이었|으로|인\s)|목표(?:다|이다|이며|이고|로|입니다)"
    r"|(?:예정|계획|목표)\s*(?:[.,。)\]…]|$)"                    # 제목·문장 끝 '출시 예정'
    r"|것으로\s*(?:예상|전망|기대)|앞두고|앞둔|내년|\bnext\s+year\b"
    r"|\b(?:plans?|aims?|expects?|intends?|hopes?)\s+to\b|\bis\s+(?:set|scheduled|expected|poised)\s+to\b", re.I)
# 완료 표지
_NOT_YET = r"(?!\s*(?:할|될|하기|예정|목표|계획))"
DONE = re.compile(
    r"(?<!기로 )(?<!기로)(?:했|하였|됐|되었)(?:다|으며|고|습니다)"
    r"|(?<!추진)(?<!계획)(?<!준비)(?<!검토)(?<!모색)(?<!협의)(?<!논의)하고\s*있(?:다|으며|습니다)"
    rf"|(?:운영|사용|판매|공급|가동)\s*중|(?:완료|달성|기록|체결|수상){_NOT_YET}|선정됐|선정되었|유치했|받았"
    r"|\b(?:raised|closed|secured|deployed|installed|launched|operates|operating|serves|served|signed|won)\b", re.I)


def parse_date(when) -> tuple[int, int | None] | None:
    """날짜 필드 → (연, 월 | None). 연도를 못 읽으면 None, 월이 1~12 밖이면 월 None."""
    m = _FIELD.match(str(when or "").strip())
    if not m:
        return None
    mo = int(m.group(2)) if m.group(2) else None
    return int(m.group(1)), (mo if mo and 1 <= mo <= 12 else None)


def _after(y: int, mo: int | None, run_date: str) -> bool:
    ry, rm = int(run_date[:4]), int(run_date[5:7])
    return y > ry if mo is None else (y, mo) > (ry, rm)


def is_future(when, run_date: str) -> bool:
    """when 이 평가 기준일보다 뒤인가 (적힌 정밀도로 비교). 날짜를 못 읽으면 False(판단하지 않음)."""
    p = parse_date(when)
    return p is not None and _after(*p, run_date)


def in_window(y: int, mo: int | None, run_date: str, months: int | None = None) -> bool:
    """(연, 월 | None) 이 [평가 기준일 − months 개월, 평가 기준일] 안인가. 연도만 있으면 기준 연도 − months/12 ~ 기준 연도."""
    months = months or recent_months()
    if _after(y, mo, run_date):
        return False
    ry, rm = int(run_date[:4]), int(run_date[5:7])
    return y >= ry - months // 12 if mo is None else (ry - y) * 12 + (rm - mo) <= months


def text_dates(text: str) -> list[tuple[int, int | None]]:
    """문장 속 사건 날짜 [(연, 월 | None)]."""
    text = text or ""
    out = []
    for m in _TEXT_YM.finditer(text):
        mo = int(m.group(2) or m.group(3))
        if 1 <= mo <= 12:
            out.append((int(m.group(1)), mo))
    out += [(int(y), None) for y in _TEXT_Y.findall(text) + _TEXT_RANGE.findall(text)]
    out += [(int(y), _MONTHS.index(mon[:3].lower()) + 1) for mon, y in _EN_YM.findall(text)]
    ym = {m.start(2) for m in _EN_YM.finditer(text)}
    out += [(int(m.group(1)), None) for m in _EN_Y.finditer(text) if m.start(1) not in ym]
    return out


def only_future_events(text: str, run_date: str, months: int | None = None) -> bool:
    """문장 속 날짜가 기준일 이후뿐이면 True (최근 사건 근거가 아님)."""
    dates = text_dates(text)
    future = [d for d in dates if _after(*d, run_date)]
    if not future or any(in_window(*d, run_date, months) for d in dates):
        return False
    return len(future) < len(dates) or not DONE.search(text or "")


def round_planned(quote: str) -> bool:
    """투자 단계 인용이 끝난 라운드가 아니라 예정·추진 중인 라운드'만' 말하는가 (끝난 라운드가 함께 적혀 있으면 False)."""
    return bool(ROUND_PLANNED.search(quote or "")) and not ROUND_DONE.search(quote or "")


def future_round(record: dict, run_date: str) -> str | None:
    """적격성 기록의 최근 라운드가 기준일 이후이거나 예정 라운드면 그 이유(G2 불인정), 아니면 None."""
    rd = str(record.get("round_date") or "")
    if is_future(rd, run_date):
        return f"최근 라운드 시점({rd})이 평가 기준일({run_date})보다 뒤 — 완료된 투자로 확인되지 않음"
    if round_planned(record.get("stage_quote") or ""):
        return "단계 인용이 예정·추진 중인 라운드 — 완료된 투자로 확인되지 않음"
    return None


def planned_only(text: str, run_date: str, months: int | None = None) -> bool:
    """예정·계획·목표 문장인데 완료 표지도, 창(최근 months 개월) 안의 사건 날짜도 없으면 True (완료된 사건이 아님)."""
    return (bool(PLANNED.search(text or "")) and not DONE.search(text or "")
            and not any(in_window(*d, run_date, months) for d in text_dates(text)))
