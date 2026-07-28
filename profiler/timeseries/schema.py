"""4-3/4-4 출력 스키마 (schema_version=1, source="timeseries").

소비 지점:
  periodicity.mem_peak  → Track 3 메모리 선제 관리 (문서화만 — 실연결 없음)
  periodicity.period_s  → 4-4 hold 산정, Agent 태깅 (문서화만 — 실연결 없음)
  termination           → Turnaround Booster 트리거 — 발화 경로 구현됨 (Exp_24):
                          term_online.py --events PATH 가 TERM_PREDICTED 이벤트를
                          JSON Lines 로 append. 소비자: Turnaround Booster (Exp_28 예정)
"""
SCHEMA_VERSION = 1

_TERM_CONF = ("LOW", "HIGH")


def build_output(logical_unit_id, periodicity, mem_peak, termination_events,
                 est_remaining_s=None):
    """분석 결과 → 4-3/4-4 통합 출력 dict."""
    term_event = termination_events[-1] if termination_events else None
    out = {
        "schema_version": SCHEMA_VERSION,
        "source": "timeseries",
        "logical_unit_id": logical_unit_id,
        "periodicity": {**periodicity, "mem_peak": mem_peak},
        "termination": {
            "terminating": term_event is not None,
            "t_signal": term_event["t_signal"] if term_event else None,
            "hold_s": term_event["hold_s"] if term_event else None,
            "confidence": term_event["confidence"] if term_event else None,
            "est_remaining_s": est_remaining_s,
            "basis": term_event["basis"] if term_event else None,
        },
        "note": "동적 시계열 관측 — 주기/임계값은 잠정, 사후 분석 기반 "
                "(온라인화는 후속 — Exp_14 report §6).",
    }
    errs = validate_output(out)
    if errs:
        raise AssertionError(f"output schema violation: {errs}")
    return out


# ---------------------------------------------------------------------------
# TERM_PREDICTED 이벤트 (Exp_24 — 4-4 온라인 종료감지 → Turnaround Booster)
#
# 발화자: term_online.py (--events PATH, JSON Lines append, 판정 순간 1줄)
# 소비자: Turnaround Booster (Exp_28 예정) — 파일 tail-follow 로 구독
#
# 한 줄 = 이벤트 1개:
#   {ts:        float  epoch 초 (발화 시각),
#    event:     "TERM_PREDICTED",
#    schema_version: 1,
#    gpu:       str    GPU uuid 또는 index (nvidia-smi --id 값),
#    pids:      [int]  판정 시점 GPU compute 프로세스 (종료 직후면 빈 리스트),
#    evidence:  {basis: "vram_release"|"idle_hold",
#                confidence: "HIGH"|"LOW",
#                t_signal: float, hold_s: float,
#                baseline_mb: int, peak_mb: int,
#                util: int, mem_mb: int,
#                periodicity: float|None}}   # 온라인 모니터는 None (주기 미산출)
# ---------------------------------------------------------------------------

_TERM_EVENT = "TERM_PREDICTED"
_TERM_BASIS = ("vram_release", "idle_hold")


def build_term_event(gpu, pids, evidence, ts=None):
    """TERM_PREDICTED 이벤트 dict 생성 (+스키마 검증)."""
    import time as _time
    ev = {
        "ts": round(_time.time() if ts is None else ts, 3),
        "event": _TERM_EVENT,
        "schema_version": SCHEMA_VERSION,
        "gpu": str(gpu),
        "pids": list(pids),
        "evidence": dict(evidence),
    }
    errs = validate_term_event(ev)
    if errs:
        raise AssertionError(f"term event schema violation: {errs}")
    return ev


def validate_term_event(ev):
    errs = []
    for k in ("ts", "event", "schema_version", "gpu", "pids", "evidence"):
        if k not in ev:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if ev["event"] != _TERM_EVENT:
        errs.append(f"event != {_TERM_EVENT}")
    if not isinstance(ev["ts"], (int, float)):
        errs.append("ts number")
    if not (isinstance(ev["pids"], list)
            and all(isinstance(p, int) for p in ev["pids"])):
        errs.append("pids [int]")
    e = ev["evidence"]
    if e.get("basis") not in _TERM_BASIS:
        errs.append("evidence.basis enum")
    if e.get("confidence") not in _TERM_CONF:
        errs.append("evidence.confidence enum")
    for k in ("t_signal", "hold_s", "baseline_mb", "peak_mb"):
        if not isinstance(e.get(k), (int, float)):
            errs.append(f"evidence.{k} number")
    return errs


def validate_output(o):
    errs = []
    for k in ("schema_version", "source", "logical_unit_id",
              "periodicity", "termination"):
        if k not in o:
            errs.append(f"missing {k}")
    if errs:
        return errs
    p = o["periodicity"]
    if not isinstance(p.get("periodic"), bool):
        errs.append("periodicity.periodic bool")
    if p.get("periodic") and not (isinstance(p.get("period_s"), (int, float))
                                  and p["period_s"] > 0):
        errs.append("periodic=True 인데 period_s 없음")
    if not p.get("periodic") and p.get("period_s") is not None:
        errs.append("periodic=False 면 period_s 는 None (억지 주기 금지)")
    if "basis" not in p:
        errs.append("periodicity.basis 누락 (근거 명시 의무)")
    t = o["termination"]
    if not isinstance(t.get("terminating"), bool):
        errs.append("termination.terminating bool")
    if t["terminating"]:
        if t.get("confidence") not in _TERM_CONF:
            errs.append("termination.confidence enum")
        if not t.get("basis"):
            errs.append("termination.basis 누락")
    return errs
