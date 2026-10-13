"""AgTech 스타트업 투자 평가 에이전트 실행 스크립트.

    python app.py                  # Supervisor 그래프로 새로 평가 (검색·LLM 을 새로 호출, 결과는 cache_fresh/ 에 저장)
    python app.py --sequential     # 예전 순차 그래프 (탐색 → 창업자 → 기술 → 시장성 → 경쟁사 → 투자 판단 → 보고서)
    python app.py --replay         # 지난 제출본 캐시(replay/)로 실행
    python app.py --offline        # 저장된 캐시만 쓰고 API 키 없이 실행 (보통 --replay 와 함께)
    python app.py --calibrate      # 평가 상한까지 평가해 동종 기준 집단(data/reference_class.json)을 만든다 (보고서 없음)
    python app.py --out DIR        # 결과를 DIR 에 쓴다 (제출용 파일은 만들지 않음)
    python app.py --graph-only     # 그래프 그림만 생성
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import warnings

warnings.filterwarnings("ignore")

from core.config import ROOT, get_config, path, require_keys
from core.config import run_date as get_run_date
from core.cost import TRACKER
from graph.builder import build_graph
from tools import web_search as web_search_mod

CALIBRATION_DIR = "outputs/calibration"
SENSITIVITY_FILE = "outputs/eval/threshold_sensitivity.json"
SCREENED_KEYS = ("name", "region", "stage", "round_date", "founded_date", "ceo", "eligible", "reason",
                 "gate_search_failed", "channels")
EVALUATION_KEYS = ("name", "region", "stage", "decision", "hold_type", "multiplier", "score100", "reasons", "flip")
ROW_KEYS = ("qid", "answer", "rationale", "quote", "evidence_ids")


def save_graph_image(app) -> str:
    """LangGraph 가 직접 그린 그래프 그림 (docs/architecture_langgraph.png)."""
    from report.mermaid import mermaid_to_png

    mmd = app.get_graph(xray=1).draw_mermaid()
    path("docs/architecture_langgraph.mmd").write_text(mmd, encoding="utf-8")
    out = path("docs/architecture_langgraph.png")
    mermaid_to_png(mmd, out)
    return str(out)


def _check_browser() -> None:
    """PDF 생성용 Chromium 이 없으면 시작할 때 멈춘다."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        if not Path(p.chromium.executable_path).exists():
            raise SystemExit("PDF 생성용 브라우저가 없습니다. 먼저 `python -m playwright install chromium` 을 실행하세요.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--replay", action="store_true",
                        help="새로 호출하지 않고 지난 제출본 캐시(replay/)로 실행")
    parser.add_argument("--sequential", action="store_true",
                        help="Supervisor 대신 예전 순차 그래프로 실행")
    parser.add_argument("--offline", action="store_true",
                        help="재현 테스트: 캐시에 없는 검색·LLM 호출이 생기면 바로 실패 (API 키 없이 실행 가능)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="재현용 캐시에 실패로 표시된 검색만 다시 시도 (검색 한도 초과로 비었던 근거를 새 키로 보강)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="투자 기준 배수(decision.threshold)를 이번 실행에서만 바꾼다. 제출 보고서를 덮어쓰지 않게 --out 과 함께 쓴다")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true",
                      help=f"보정 실행: 투자여도 멈추지 않고 평가 상한까지 평가 → 기준 집단 파일 생성, 실행 기록은 {CALIBRATION_DIR}/")
    mode.add_argument("--out", metavar="DIR", default=None,
                      help="시나리오 실행: 결과를 DIR 에 쓰고 제출용 파일(RAG-Output_…)은 만들지 않는다")
    return parser.parse_args(argv)


def apply_options(args: argparse.Namespace, cfg) -> str:
    """실행 옵션을 설정에 반영하고 실행 모드(main | calibrate | scenario)를 돌려준다."""
    if args.replay:
        cfg["cache"]["dir"] = "replay"
    if args.threshold is not None:
        cfg["decision"]["threshold"] = args.threshold
    if args.calibrate:
        cfg["workflow"]["stop_on_invest"] = False
        cfg["workflow"]["calibrate"] = True
        cfg["report"]["output_dir"] = CALIBRATION_DIR
        return "calibrate"
    if args.out:
        cfg["report"]["output_dir"] = args.out
        cfg["report"]["scenario"] = True
        return "scenario"
    return "main"


def search_notice(args: argparse.Namespace, cfg) -> str:
    """검색 공급자 안내 한 줄."""
    names = " → ".join(p.capitalize() for p in web_search_mod.active_providers()) or "없음"
    if args.offline or (args.replay and not args.retry_failed):
        return f"   검색: 재현용 캐시({cfg.cache.dir}/)에서 읽음 — 검색 키·공급자와 무관 (공급자 {names})"
    msg = f"   검색 공급자: {names} · 요청 제한 {cfg.search.get('timeout_sec', 30)}초"
    if web_search_mod.active_providers() == ["tavily"]:
        msg += (" — SERPER_API_KEY 가 없어 모든 검색을 Tavily 로 보냅니다. 새 평가는 검색이 수백 번이라 오래 걸리고"
                " Tavily 무료 한도를 많이 씁니다 (README '실행이 느리거나 멈춘 것 같을 때')")
    return msg


def initial_state(cfg, run_date: str, run_id: str = "") -> dict:
    return {"domain": cfg.domain.name, "run_date": run_date, "run_id": run_id, "registry": {}, "discovery_rounds": 0,
            "iterations": 0, "queue": [], "seen": [], "evaluations": [], "screened": [], "log": [], "rag_traces": []}


def invoke_config(run_id: str, mode: str, orchestration: str, recursion_limit: int) -> dict:
    """LangGraph 실행 설정. run_id 를 LangSmith metadata·태그와 checkpoint thread_id 에 같이 넣어 State·외부 기록을 잇는다.
    LangSmith 기록은 .env 의 LANGSMITH_TRACING=true·LANGSMITH_API_KEY 가 있을 때만 남는다."""
    return {"recursion_limit": recursion_limit, "run_name": f"agtech-{orchestration}-{mode}",
            "tags": [orchestration, mode], "metadata": {"run_id": run_id},
            "configurable": {"thread_id": run_id}}


def _evaluation_log(e: dict) -> dict:
    sc = e.get("scorecard") or {}
    return {**{k: e.get(k) for k in EVALUATION_KEYS},
            "roi": sc.get("roi"), "criteria": e.get("criteria", sc.get("criteria")),
            "rows": [{k: r.get(k) for k in ROW_KEYS} for r in sc.get("rows", [])]}


def build_run_log(state: dict, cfg, *, mode: str, run_date: str, elapsed: float) -> dict:
    return {
        "run_id": state.get("run_id"), "run_date": run_date, "elapsed_sec": elapsed, "mode": mode,
        "threshold": cfg.decision.threshold,
        "end_reason": state.get("end_reason"),
        "models": dict(cfg.models), "embedding": cfg.embedding.model,
        "discovery_rounds": state.get("discovery_rounds"), "evaluated": state.get("iterations"),
        "screened": [{k: r.get(k) for k in SCREENED_KEYS} for r in state.get("screened", [])],
        "evaluations": [_evaluation_log(e) for e in state.get("evaluations", [])],
        "report": state.get("report") or {}, "rag_traces": state.get("rag_traces", []), "log": state.get("log", []),
        "sources_collected": len(state.get("registry", {})),
        "failed_searches": sorted({(f["agent"], f["query"]) for f in getattr(web_search_mod, "FAILED_QUERIES", [])}),
        "llm_cost": TRACKER.summary(),
    }


def _write_json(rel: str, data: dict) -> None:
    # 로컬 절대경로 대신 저장소 기준 상대경로
    text = json.dumps(data, ensure_ascii=False, indent=2).replace(str(ROOT) + "/", "")
    path(rel).write_text(text, encoding="utf-8")


def v1_evaluated_names() -> list[str] | None:
    """v1-safe 태그의 실행 기록에서 평가한 후보 이름. 태그가 없으면 None."""
    try:
        out = subprocess.run(["git", "show", "v1-safe:outputs/run_log.json"], cwd=ROOT, capture_output=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return [e["name"] for e in json.loads(out.stdout.decode("utf-8")).get("evaluations", [])]


def check_v1_members(names: list[str]) -> bool | None:
    """보정 실행의 평가 후보가 v1-safe 태그의 평가 후보와 같은지."""
    v1 = v1_evaluated_names()
    if v1 is None:
        print("[캐시 보존 확인] v1-safe 태그를 찾지 못해 비교를 건너뜁니다.")
        return None
    added, missing = sorted(set(names) - set(v1)), sorted(set(v1) - set(names))
    if not added and not missing:
        print(f"[캐시 보존 확인] 평가 후보 {len(set(names))}곳이 v1 평가 후보와 같습니다.")
        return True
    print(f"[경고] 평가 후보가 v1 과 다릅니다 (추가 {added}, 빠짐 {missing}). 발굴·관문 캐시가 바뀌었는지 확인하세요.")
    return False


def finish_calibration(state: dict, cfg, run_date: str) -> dict:
    """보정 실행 뒤: 동종 기준 집단 파일과 기준 배수 민감도를 쓴다."""
    from agents.decision import write_reference_class, write_threshold_sensitivity

    evaluations = state.get("evaluations", [])
    ref_file = cfg.decision.reference_file
    ref = write_reference_class(evaluations, str(path(ref_file)), run_date)
    sens = write_threshold_sensitivity(str(path(ref_file)), str(path(SENSITIVITY_FILE)))
    n, need = ref.get("n") or 0, cfg.decision.reference_min_n
    enough = n >= need
    print(f"기준 집단: {n}곳 → {ref_file}")
    if not enough:
        print(f"[경고] 기준 집단 {n}곳 < 최소 {need}곳(decision.reference_min_n). 본 실행은 fallback(동종 평균 신호 0)으로 계산됩니다.")
    print(f"기준 배수별 투자 추천 수: {json.dumps(sens.get('invest_count_at', {}), ensure_ascii=False)} → {SENSITIVITY_FILE}")
    same = check_v1_members([m["name"] for m in ref.get("members") or []])
    return {"reference_file": ref_file, "reference_n": n, "reference_enough": enough, "sensitivity_file": SENSITIVITY_FILE,
            "invest_count_at": sens.get("invest_count_at"), "same_as_v1_evaluated": same}


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    if args.retry_failed:
        os.environ["SEARCH_RETRY_FAILED"] = "1"
    if args.offline:
        os.environ["REPLAY_OFFLINE"] = "1"
    _check_browser()

    cfg = get_config()
    base_out = cfg.report.output_dir
    mode = apply_options(args, cfg)
    supervised = not args.sequential
    if not (args.offline or args.graph_only):
        require_keys()
    if supervised:
        from graph.supervisor import build_supervisor_graph, supervisor_recursion_limit

        app = build_supervisor_graph()
    else:
        app = build_graph()
    if args.graph_only:
        print(save_graph_image(app))
        return

    t0 = time.time()
    run_date = get_run_date()
    meta = path(f"{cfg.cache.dir}/run_meta.json")
    if not meta.exists():  # 같은 캐시로 다시 돌리면 같은 기준일
        meta.write_text(json.dumps({"run_date": run_date}), encoding="utf-8")
    print(f"== AgTech 스타트업 투자 평가 시작 ({run_date}, 모드 {mode}, 기준 배수 {cfg.decision.threshold}, "
          f"생성 {cfg.models.generator}, 판정 {cfg.models.judge}, 임베딩 {cfg.embedding.model}, 캐시 {cfg.cache.dir}/) ==")
    print(search_notice(args, cfg))
    recursion_limit = supervisor_recursion_limit() if supervised else cfg.workflow.recursion_limit
    run_id = uuid.uuid4().hex[:12]
    orchestration = "supervisor" if supervised else "sequential"
    print(f"   run_id {run_id} (run_log·LangSmith metadata 의 연결 키)")
    state = app.invoke(initial_state(cfg, run_date, run_id), invoke_config(run_id, mode, orchestration, recursion_limit))
    elapsed = round(time.time() - t0, 1)

    out_dir = cfg.report.output_dir
    run_log = build_run_log(state, cfg, mode=mode, run_date=run_date, elapsed=elapsed)
    run_log["orchestration"] = orchestration
    if supervised:
        run_log["supervisor"] = {"steps": state.get("supervisor_steps"), "reviews": state.get("supervisor_reviews", {}),
                                 "final_review": state.get("supervisor_final_review", {}),
                                 "excluded": state.get("supervisor_excluded", [])}
        run_log["report_evaluation"] = {"revisions": state.get("report_revisions", 0),
                                        **(state.get("report_evaluation") or {})}
    _write_json(f"{out_dir}/run_log.json", run_log)
    cost = TRACKER.summary()
    if cost.get("api_calls"):
        with open(path(f"{base_out}/cost_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_date": run_date, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "elapsed_sec": elapsed, "fresh": not args.replay, "mode": mode, **cost},
                               ensure_ascii=False) + "\n")
    print(f"== 완료 ({elapsed}초, 종료 사유 {state.get('end_reason')}, LLM {cost}) ==")

    report = state.get("report") or {}
    if mode == "calibrate":
        run_log["calibration"] = finish_calibration(state, cfg, run_date)
        _write_json(f"{out_dir}/run_log.json", run_log)
    elif report.get("submit_pdf") or report.get("pdf"):
        print(f"보고서: {report.get('submit_pdf') or report.get('pdf')}")
        if report.get("checks"):
            print(f"검증: {json.dumps(report['checks'], ensure_ascii=False)}")
    else:
        print(f"보고서를 만들지 못했습니다. {out_dir}/run_log.json 을 확인하세요.")


if __name__ == "__main__":
    main()
