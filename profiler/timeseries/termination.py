"""4-4 종료 임박 감지 — 활용률 급감 + 4-3 주기 결합.

핵심: "iteration 간 정상 휴지"와 "진짜 종료"의 구분.
  - 주기 있는 워크로드: 유휴 지속 ≥ PERIOD_K × 주기 일 때만 종료 신호
    (주기 내 휴지는 정의상 주기보다 짧다).
  - 주기 없는 워크로드(버스트 추론): 관측된 유휴 gap 이력 기반
    hold = GAP_K × 최대 관측 gap (이력 부족하면 confidence LOW).
  - 주기도 이력도 없으면 naive(고정 hold) — 오탐 가능성을 confidence LOW 로 표기.
"""
import numpy as np

DEFAULT_IDLE_UTIL = 10       # 이 미만이면 유휴 (%)
NAIVE_HOLD_S = 0.5           # naive 고정 hold (오탐 대조용)
PERIOD_K = 3                 # 주기 결합: hold = K × period
GAP_K = 2                    # 이력 결합: hold = K × 최대 관측 gap
MIN_HOLD_S = 1.0
MIN_GAP_SAMPLES = 5          # 이력 기반 hold 의 신뢰 최소 gap 표본 수


def hold_from_gaps(idle_gaps_s):
    """비주기 워크로드용: 활동 구간에서 관측된 유휴 gap 이력 → hold."""
    gaps = [g for g in idle_gaps_s if g > 0]
    if not gaps:
        return MIN_HOLD_S, "LOW"
    hold = max(MIN_HOLD_S, GAP_K * max(gaps))
    conf = "HIGH" if len(gaps) >= MIN_GAP_SAMPLES else "LOW"
    return hold, conf


MEM_RELEASE_FRAC = 0.1       # 활성 중앙값 대비 이 비율 미만이면 VRAM 해제로 판정


def detect_termination(t, util, idle_util=DEFAULT_IDLE_UTIL,
                       period_s=None, idle_gaps_s=None, hold_s=None,
                       mem_used_mb=None):
    """활용률(+선택: VRAM) 시계열 스캔 → 종료 신호 이벤트 리스트.

    hold 결정 우선순위: 명시 hold_s > 주기(PERIOD_K×period) > gap 이력 > naive.
    ★ 첫 활동(util ≥ idle_util) 관측 후에만 무장 — 시작 전(모델 로드 등)
      유휴를 종료로 오인하지 않음 (Exp_14 실측: 로드 구간 오탐).
    ★ mem_used_mb 제공 시: 활성 중앙값 대비 MEM_RELEASE_FRAC 미만으로
      떨어지면 즉시 발화 (VRAM 해제 = 확정 신호, hold 불필요 — Exp_14 실측:
      util-hold 보다 빠르고 teardown 의 util 스파이크에 영향받지 않음).
    유휴 stretch 당 1회 발화. 이벤트:
      {t_signal, idle_since, hold_s, confidence, basis, kind}
    """
    if hold_s is not None:
        conf, basis = "HIGH", f"명시 hold={hold_s}s"
    elif period_s:
        hold_s = max(PERIOD_K * period_s, MIN_HOLD_S)
        conf, basis = "HIGH", (f"4-3 주기 {period_s}s × {PERIOD_K} — "
                               f"주기 내 휴지는 주기를 넘지 못함")
    elif idle_gaps_s is not None:
        hold_s, conf = hold_from_gaps(idle_gaps_s)
        basis = f"비주기 — 관측 gap 이력 최대 × {GAP_K}"
    else:
        hold_s, conf = NAIVE_HOLD_S, "LOW"
        basis = "naive 고정 hold — 휴지/종료 구분 근거 없음 (오탐 가능)"

    t = np.asarray(t, dtype=float)
    u = np.asarray(util, dtype=float)
    m = None if mem_used_mb is None else np.asarray(mem_used_mb, dtype=float)
    events = []
    idle_since = None
    fired = armed = mem_fired = False
    active_mem = []
    for i, (ti, ui) in enumerate(zip(t, u)):
        if ui >= idle_util:
            armed = True
            idle_since, fired = None, False
            if m is not None:
                active_mem.append(m[i])
            continue
        if not armed:
            continue                       # 활동 시작 전 유휴 — 무시
        if idle_since is None:
            idle_since, fired = ti, False
        if not fired and ti - idle_since >= hold_s:
            events.append({"t_signal": float(ti),
                           "idle_since": float(idle_since),
                           "hold_s": round(float(hold_s), 3),
                           "confidence": conf, "basis": basis,
                           "kind": "util_hold"})
            fired = True
        if (m is not None and not mem_fired and len(active_mem) >= 3
                and m[i] < MEM_RELEASE_FRAC * float(np.median(active_mem))):
            events.append({"t_signal": float(ti),
                           "idle_since": float(idle_since),
                           "hold_s": 0.0, "confidence": "HIGH",
                           "basis": f"VRAM 해제 (활성 중앙값 "
                                    f"{np.median(active_mem):.0f}→{m[i]:.0f}MB)",
                           "kind": "mem_release"})
            mem_fired = True
    return events


def idle_gaps(t, util, idle_util=DEFAULT_IDLE_UTIL):
    """활동 구간 내 유휴 gap 길이 목록 (이력 기반 hold 산출용).

    마지막 gap 이 시계열 끝까지 이어지면(종료 후 구간) 제외한다.
    """
    t = np.asarray(t, dtype=float)
    u = np.asarray(util, dtype=float)
    gaps, start = [], None
    for i, (ti, ui) in enumerate(zip(t, u)):
        if ui < idle_util:
            if start is None:
                start = ti
        else:
            if start is not None:
                gaps.append(float(ti - start))
                start = None
    return gaps
