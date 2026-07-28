"""phase_online — 온라인 국면 감지 (Trace Analyzer 4-2/4-4 온라인화, Exp_19).

입력: kernel_tracer(CUPTI Activity, Exp_14 스트리밍 모드 — 작은 버퍼 회전 +
완료 버퍼만 주기 플러시)가 실행 중 계속 append 하는 트레이스 파일.
tail-follow 로 새 레코드를 읽어 슬라이딩 윈도(GPU 시간 기준)에 유지하고,
주기적으로 국면을 판정한다.

국면 라벨 (Exp_13/17 실측 시그니처 근거):
  PREFILL         GEMM 지배(C/M ≥ 0.55) + KV-append(Cat) 부재   [Exp_13: C/M 0.912, cat 0]
  DECODE          gemv/splitkv 지배 (memory-bound b=1 decode)   [Exp_13: gemv 81%+cat 7.9%]
  DECODE_BATCHED  GEMM 지배 + KV-append(Cat) 존재 = b≥4 반전    [Exp_17: C/M 0.62~0.65, cat 존재]
  MIXED / IDLE    중간 / 윈도 내 커널 없음·희소

판정 근거는 전부 커널명 패턴(kernel_class.patterns 재사용) + 윈도 시간 점유 —
값을 만들지 않고 근거 부족 시 confidence LOW (Exp_13 제약 계승).
"""
import json
import os
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kernel_class.patterns import classify_kernel_name  # noqa: E402

# 판정 임계 (기존 트레이스 4종 오프라인 보정 — calib_offline.py 실측 근거)
#   prefill(use_cache=True):  cm 0.911, cat_share 0.025  ← cat 은 prefill 에도 존재!
#   decode b=1:               gemv_share 0.87
#   decode b=4/8:             cm 0.645/0.663, cat_share 0.213/0.317
# → GEMM 지배 윈도의 prefill vs batched-decode 판별은 cat_share(0.025 vs 0.21+,
#   자릿수 차이) + cm(0.91 vs 0.62~0.66) 이중 기준.
GEMV_SHARE_DECODE = 0.30      # gemv+splitkv 시간 점유 ≥ 0.30 → DECODE
CAT_SHARE_MARKER = 0.08       # KV-append(Cat) 시간 점유 ≥ 8% → batched-decode 마커
CM_PREFILL = 0.80             # C/(C+M) ≥ 0.80 → cat 여부 무관 PREFILL (실측 0.911)
CM_HIGH = 0.55                # C/(C+M) ≥ 0.55 → compute 지배
CM_LOW = 0.45
MIN_KERNELS = 30              # 윈도 최소 커널 수 (미만 = 신뢰 불가/IDLE 판단 재료)


class _NameCache:
    """mangled 커널명 → (class, is_gemv, is_cat) 캐시 (유니크 커널 수십 개 수준)."""

    def __init__(self):
        self.cache = {}

    def get(self, name):
        e = self.cache.get(name)
        if e is None:
            low = name.lower()
            cls, _ = classify_kernel_name(name)
            e = (cls,
                 ("gemv" in low) or ("splitkv" in low),
                 ("batchedcopy" in low) or ("catarray" in low))
            self.cache[name] = e
        return e


def classify_window(records, name_cache):
    """records = [(start_ns, dur_ns, name)] → (label, confidence, evidence)."""
    c_ns = m_ns = u_ns = gemv_ns = cat_ns = 0
    for _, dur, name in records:
        cls, is_gemv, is_cat = name_cache.get(name)
        if cls == "COMPUTE":
            c_ns += dur
        elif cls == "MEMORY":
            m_ns += dur
        else:
            u_ns += dur
        if is_gemv:
            gemv_ns += dur
        if is_cat:
            cat_ns += dur
    tot = c_ns + m_ns
    n = len(records)
    ev = {"n_kernels": n, "c_ms": round(c_ns / 1e6, 2), "m_ms": round(m_ns / 1e6, 2),
          "uncertain_ms": round(u_ns / 1e6, 2)}
    if n < MIN_KERNELS or tot == 0:
        return "IDLE", "LOW", ev
    cm = c_ns / tot
    gemv_share = gemv_ns / tot
    cat_share = cat_ns / tot
    ev.update(cm_ratio=round(cm, 3), gemv_share=round(gemv_share, 3),
              cat_share=round(cat_share, 4))
    if gemv_share >= GEMV_SHARE_DECODE:
        conf = "HIGH" if gemv_share >= 0.5 else "LOW"
        return "DECODE", conf, ev
    if cm >= CM_HIGH:
        if cm < CM_PREFILL and cat_share >= CAT_SHARE_MARKER:
            conf = "HIGH" if n >= 100 else "LOW"
            return "DECODE_BATCHED", conf, ev
        conf = "HIGH" if (cm >= 0.70 or cat_share < CAT_SHARE_MARKER) and n >= 100 else "LOW"
        return "PREFILL", conf, ev
    if cm <= CM_LOW:
        conf = "HIGH" if cm <= 0.35 else "LOW"
        return "DECODE", conf, ev
    return "MIXED", "LOW", ev


class PhaseDetector:
    """슬라이딩 윈도(GPU ns) + 판정 히스테리시스(연속 confirm 회) 국면 감지기.

    사용: d = PhaseDetector(); d.add_line(line) 반복; 주기적으로 d.judge() 호출 →
    판정 dict(라벨 전환 시 transition 필드 포함) 반환.
    """

    def __init__(self, window_s=1.0, confirm=2, idle_gap_s=0.5):
        self.window_ns = int(window_s * 1e9)
        self.idle_gap_ns = int(idle_gap_s * 1e9)
        self.confirm = confirm
        self.records = []                   # (start_ns, dur_ns, name) — start 정렬 근사
        self.names = _NameCache()
        self.max_ns = 0
        self.stable_label = None
        self._cand = None
        self._cand_n = 0
        self.n_seen = self.n_zero_start = self.n_dropped = 0

    def add_line(self, line):
        if line.startswith("K|"):
            p = line.split("|", 8)
            if len(p) < 9:
                return
            start, dur = int(p[1]), int(p[2])
            self.n_seen += 1
            if start == 0:                  # Exp_14 손상 레코드 가드
                self.n_zero_start += 1
                return
            self.records.append((start, dur, p[8]))
            if start > self.max_ns:
                self.max_ns = start
        elif line.startswith("#dropped"):
            try:
                self.n_dropped += int(line.split("|")[1])
            except (IndexError, ValueError):
                self.n_dropped += 1

    def _trim(self):
        lo = self.max_ns - self.window_ns
        self.records = [r for r in self.records if r[0] >= lo]

    def judge(self, wall_ts=None):
        """현재 윈도 판정. 반환 dict 는 events.jsonl 한 줄 스키마."""
        wall_ts = wall_ts if wall_ts is not None else time.time()
        self._trim()
        # 최근 idle_gap 동안 새 레코드 없으면(플러시 지연과 구분 불가) 마지막 판정 유지 + IDLE 후보
        label, conf, ev = classify_window(self.records, self.names)
        out = {"t": round(wall_ts, 3), "label_raw": label, "confidence": conf,
               "evidence": ev, "stable": self.stable_label}
        if label != self.stable_label:
            if label == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = label, 1
            if self._cand_n >= self.confirm:
                out["transition"] = {"from": self.stable_label, "to": label,
                                     "t": round(wall_ts, 3)}
                self.stable_label = label
                self._cand, self._cand_n = None, 0
        else:
            self._cand, self._cand_n = None, 0
        out["stable"] = self.stable_label
        return out

    def health(self):
        return {"records_seen": self.n_seen, "zero_start": self.n_zero_start,
                "dropped": self.n_dropped, "unique_kernels": len(self.names.cache)}


def follow_trace(path, poll_s=0.05, stop_check=None):
    """트레이스 파일 tail-follow 제너레이터 — 새 줄을 yield, stop_check() 참이면 종료."""
    f = None
    buf = ""
    while f is None:
        try:
            f = open(path, "r")
        except FileNotFoundError:
            if stop_check and stop_check():
                return
            time.sleep(poll_s)
    while True:
        chunk = f.read()
        if chunk:
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line
        else:
            if stop_check and stop_check():
                return
            time.sleep(poll_s)


def build_output(logical_unit_id, events, health, window_s, confirm):
    """4-2/4-4 공통 규약(schema_version=1, source, confidence, basis) 준수 요약."""
    transitions = [e["transition"] for e in events if "transition" in e]
    return {
        "schema_version": 1, "source": "phase_online",
        "logical_unit_id": logical_unit_id,
        "window_s": window_s, "confirm": confirm,
        "transitions": transitions,
        "n_events": len(events), "health": health,
        "basis": "CUPTI streaming(kernel_tracer, 완료버퍼만) + 커널명 패턴(kernel_class) 윈도 점유",
    }
