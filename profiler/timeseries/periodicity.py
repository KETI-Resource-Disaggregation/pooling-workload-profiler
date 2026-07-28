"""4-3 주기성 검출 — 커널 launch 시계열의 autocorrelation + FFT 교차 확인.

★ 원칙 (Exp_14 제약): 주기 없는 워크로드에 억지 주기 금지 — strength 임계
  미달이면 periodic=False 로 정직하게 반환. 임계값은 잠정(관측 6구성 기반)이며
  근거를 반환값 basis 에 명시한다.
"""
import numpy as np

# 잠정 임계 (Exp_14 실측: 학습/decode strength 0.5~0.9, 버스트 <0.3 — report §2)
PERIODIC_MIN_STRENGTH = 0.3
FFT_AGREE_TOL = 0.2          # ACF 주기와 FFT 주기의 상대 오차 허용
DEFAULT_BIN_S = 0.005


MAX_BINS = 20_000_000     # 이상 타임스탬프(예: start=0 레코드)로 인한 폭주 방지


def bin_events(times_s, dt=DEFAULT_BIN_S):
    """이벤트 시각 리스트 → 고정 폭 bin 카운트 시계열. 반환 (x, t0).

    ★ 타임스탬프 0/음수는 무효 레코드(CUPTI FORCED 플러시의 미완료 레코드
    등 — Exp_14 실측)로 보고 제외한다. span 이 MAX_BINS 를 넘으면 입력
    오염으로 판단하고 거부 (TB 급 배열 할당 사고 방지 — Exp_14 실사고).
    """
    t = np.asarray(times_s, dtype=float)
    t = t[t > 0]
    if t.size < 2:
        raise ValueError("유효 이벤트 2개 미만 — 시계열 불성립")
    t0 = t.min()
    span_bins = int((t.max() - t0) / dt) + 1
    if span_bins > MAX_BINS:
        raise ValueError(
            f"시계열 span {span_bins} bins > {MAX_BINS} — 타임스탬프 오염 의심")
    idx = ((t - t0) / dt).astype(int)
    x = np.zeros(span_bins)
    np.add.at(x, idx, 1.0)
    return x, t0


def _autocorr(x):
    """FFT 기반 정규화 autocorrelation (lag 0..n-1)."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0] if acf[0] > 0 else acf


def detect_period(x, dt, min_lag_s=0.01, max_lag_s=10.0):
    """bin 시계열 → 주기 판정.

    방법: ACF 의 첫 zero-crossing 이후 구간에서 "최대 peak 의 80% 이상인
    가장 이른 peak" 채택 — 주기 신호의 ACF 는 주기 배수마다 비슷한 높이의
    peak 을 갖므로 전역 최대만 취하면 임의 배수를 주기로 오인한다
    (기본 주파수 우선). zero-crossing 시작은 단일 버스트의 단거리
    자기상관 오인 방지. FFT 지배 주파수로 교차 확인 (1x/0.5x/2x 허용).
    """
    acf = _autocorr(x)
    n = len(acf)
    lo = max(1, int(min_lag_s / dt))
    hi = min(n - 1, int(max_lag_s / dt))
    result = {"periodic": False, "period_s": None, "strength": 0.0,
              "fft_period_s": None, "fft_agrees": None,
              "method": "acf+fft",
              "basis": f"ACF zero-crossing 이후 최대 peak, 임계 strength≥"
                       f"{PERIODIC_MIN_STRENGTH}(잠정), bin={dt*1000:.0f}ms"}
    if hi <= lo:
        result["basis"] += " — 시계열이 탐색 lag 범위보다 짧음"
        return result

    # 첫 zero-crossing (없으면 min_lag 사용)
    neg = np.where(acf[lo:hi] < 0)[0]
    start = (neg[0] + lo) if neg.size else lo
    if start >= hi:
        result["basis"] += " — zero-crossing 이후 탐색 구간 없음 (비주기)"
        return result

    seg = acf[start:hi]
    m = float(seg.max())
    # 국소 최대 중 최대 높이의 80% 이상인 가장 이른 peak (기본 주파수)
    k = int(np.argmax(seg)) + start
    if m > 0 and len(seg) > 2:
        interior = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
                            & (seg[1:-1] >= 0.8 * m))[0]
        if interior.size:
            k = int(interior[0]) + 1 + start
    strength = float(acf[k])
    period = k * dt

    # FFT 교차 확인 (같은 lag 범위의 주파수 대역)
    xz = x - x.mean()
    ps = np.abs(np.fft.rfft(xz)) ** 2
    freqs = np.fft.rfftfreq(len(xz), dt)
    band = (freqs >= 1.0 / max_lag_s) & (freqs <= 1.0 / max(min_lag_s, dt))
    fft_period = None
    agrees = None
    if band.any():
        fft_period = float(1.0 / freqs[band][np.argmax(ps[band])])
        agrees = any(abs(fft_period - period * c) <= FFT_AGREE_TOL * period * c
                     for c in (1.0, 0.5, 2.0))

    result.update({
        "strength": round(strength, 3),
        "fft_period_s": round(fft_period, 4) if fft_period else None,
        "fft_agrees": agrees,
    })
    if strength >= PERIODIC_MIN_STRENGTH:
        result["periodic"] = True
        result["period_s"] = round(period, 4)
    return result


def detect_period_scan(candidates, dt=DEFAULT_BIN_S, min_lag_s=0.01,
                       max_lag_s=10.0):
    """여러 후보 시계열 중 최고 strength 의 주기 채택.

    candidates: {label: 이벤트 시각 리스트}. 용도: 전체 커널 카운트가 평탄한
    워크로드(decode 처럼 균일 파이프라인)에서 시그니처 커널(step 당 고정
    횟수 발화)의 서브 시계열로 주기를 잡는다 — Exp_14 실측: decode 전체
    시계열 strength 0.09, 시그니처 서브 시계열로 주기 검출.
    ★ 억지 주기 아님: 후보가 몇 개든 최종 strength 가 임계(0.3)를 넘어야
    periodic=True — 비주기 워크로드는 어떤 서브 시계열도 임계 미달.
    ★ 선택 규칙(잠정, Exp_14 실측 근거):
      1) primary(전체 스트림 'all')가 주기면 그것을 채택 — 워크로드 전체
         리듬이 1순위. (개별 커널 서브 시계열은 레이어 수준 미세 주기나
         격-step 발화(2×주기)를 제 주기로 갖는 경우가 있어 단독 신뢰 불가)
      2) 아니면 주기 인정된 서브 후보들의 주기를 15% 허용으로 군집화해
         "가장 많은 커널이 합의한 주기" 채택 (step 마다 발화하는 커널이
         다수 — 격step/서브구조 커널은 소수라 걸러짐).
      3) 주기 후보가 없으면 최대 strength 결과를 periodic=False 로 반환.
    """
    primary = "all"
    results, best_any = [], None
    for label, times in candidates.items():
        try:
            x, _ = bin_events(times, dt)
        except ValueError:
            continue
        span = len(x) * dt
        r = detect_period(x, dt, min_lag_s, min(max_lag_s, span / 2))
        r["series"] = label
        results.append(r)
        if best_any is None or r["strength"] > best_any["strength"]:
            best_any = r
    if best_any is None:
        raise ValueError("유효한 후보 시계열 없음")

    prim = next((r for r in results if r["series"] == primary), None)
    if prim is not None and prim["periodic"]:
        return prim

    periodic = [r for r in results if r["periodic"]]
    if periodic:
        # 주기값 군집화(±15%) → 최다 합의 군집, 군집 내 최고 strength
        clusters = []
        for r in sorted(periodic, key=lambda r: r["period_s"]):
            for c in clusters:
                if abs(r["period_s"] - c[0]["period_s"]) <= 0.15 * c[0]["period_s"]:
                    c.append(r)
                    break
            else:
                clusters.append([r])
        top = max(clusters, key=len)
        best = max(top, key=lambda r: r["strength"])
        best["consensus_n"] = len(top)
        return best
    return best_any


def mem_peak_analysis(t, alloc_mb, period_s, t_now=None):
    """주기 내 메모리 피크의 반복성 → 피크 크기·위상·다음 피크 예측.

    t: 샘플 시각(s), alloc_mb: 할당 메모리 시계열, period_s: 4-3 검출 주기.
    각 주기 창의 최대값을 피크로 보고, (a) 피크 크기 안정성, (b) 피크 간격의
    주기 일치, (c) 마지막 피크 + 주기 = 다음 피크 예측을 산출.
    예측 검증: 각 피크를 "직전 피크 + 주기"로 예측했을 때 오차 중앙값.
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(alloc_mb, dtype=float)
    if period_s is None or t.size < 10:
        return None
    t0, t1 = t.min(), t.max()
    n_cycles = int((t1 - t0) / period_s)
    if n_cycles < 3:
        return None
    peaks_t, peaks_v = [], []
    for k in range(n_cycles):
        m = (t >= t0 + k * period_s) & (t < t0 + (k + 1) * period_s)
        if not m.any():
            continue
        i = np.argmax(a[m])
        peaks_t.append(float(t[m][i]))
        peaks_v.append(float(a[m][i]))
    if len(peaks_t) < 3:
        return None
    peaks_t, peaks_v = np.array(peaks_t), np.array(peaks_v)
    pred_err = np.abs(np.diff(peaks_t) - period_s)          # 직전 피크+주기 예측 오차
    phase = ((peaks_t - t0) % period_s) / period_s
    now = t_now if t_now is not None else t1
    return {
        "n_peaks": len(peaks_t),
        "peak_mb_median": round(float(np.median(peaks_v)), 1),
        "peak_mb_cv": round(float(np.std(peaks_v) / np.mean(peaks_v)), 4),
        "phase_frac_median": round(float(np.median(phase)), 3),
        "pred_err_ms_median": round(float(np.median(pred_err)) * 1000, 1),
        "next_peak_eta_s": round(float(peaks_t[-1] + period_s - now), 3),
        "basis": "주기 창별 최대값 반복 관측 — 다음 피크 = 마지막 피크 + 주기",
    }
