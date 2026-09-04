#!/usr/bin/env python3
"""4-4 온라인 종료 감지 모니터 (외부 프로세스, nvidia-smi 100ms 폴링).

Exp_19 `term_monitor.py`의 승격본 (Exp_24). 판정 규칙·임계값·폴링 주기는
Exp_19 실측으로 검증된 원본 값 그대로 — 변경 금지:
  · 무장: 첫 활동(util≥20% 2연속 or VRAM 상승>500MB) 후에만 판정 (로드 유휴 오탐 차단)
  · 활동: util≥5% or VRAM > baseline+200MB
  · 신호 A (VRAM 해제): 무장 후 VRAM 이 baseline + 0.2×(peak-baseline) 아래로 → terminating
  · 신호 B (유휴 hold): 활동 gap > max(1s, 2×관측 최대 gap) → terminating (LOW)
  · teardown util 스파이크는 VRAM 신호와 무관 (Exp_14 함정 반영)

오프라인판(`termination.py`)과의 관계: 판정 철학(무장→VRAM 해제=확정 신호,
유휴 hold=보조)은 동일하나 기준식이 다르다 — 오프라인은 커널 시계열 기반
(활성 중앙값×MEM_RELEASE_FRAC), 온라인은 nvidia-smi 폴링 기반(baseline/peak).
단일화는 침습적이라 보류 (Exp_24 report §5 이연 항목).

이벤트 발화 (Exp_24 신설, schema.TERM_EVENT_SCHEMA):
  --events PATH 지정 시 종료 판정 순간 JSON Lines 1줄 append:
    {ts, event:"TERM_PREDICTED", gpu, pids, evidence:{...}}
  소비자: Turnaround Booster (Exp_28 예정).

usage: term_online.py <gpu_uuid> <out.jsonl> <stop_file> [--events events.jsonl]
매 샘플 기록 + terminating 이벤트 즉시 기록(epoch). stop_file 로 종료.
"""
import json
import os
import subprocess
import sys
import time

POLL_S = 0.1                 # Exp_19 원본 100ms — 변경 금지
ARM_UTIL = 20                # 무장: util≥20% 2연속
ARM_STREAK = 2
ARM_MEM_MB = 500             # 무장: VRAM 상승 >500MB
ACTIVE_UTIL = 5              # 활동: util≥5%
ACTIVE_MEM_MB = 200          # 활동: VRAM > baseline+200MB
RELEASE_FRAC = 0.2           # 신호 A: baseline + 0.2×(peak-baseline) 미만
HOLD_MIN_S = 1.0             # 신호 B: max(1s, 2×max_gap)
HOLD_GAP_K = 2


def gpu_pids(uuid):
    """해당 GPU 의 compute 프로세스 pid 목록 (종료 직후면 빈 리스트일 수 있음)."""
    q = subprocess.run(
        ["nvidia-smi", f"--id={uuid}", "--query-compute-apps=pid",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return [int(p) for p in q.splitlines() if p.strip().isdigit()]


def emit_term_event(events_path, gpu, evidence):
    """TERM_PREDICTED 이벤트 1줄 append (schema.validate_term_event 준수)."""
    try:
        from . import schema           # 패키지로 import 된 경우
    except ImportError:
        import schema                  # 스크립트 직접 실행 (원본 사용 방식)
    ev = schema.build_term_event(gpu=gpu, pids=gpu_pids(gpu), evidence=evidence)
    with open(events_path, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return ev


def monitor(uuid, out_path, stop_file, events_path=None):
    out = open(out_path, "w")

    baseline_mb = None
    peak_mb = 0
    armed = False
    hi_util_streak = 0
    last_active = None
    max_gap = 0.0
    signaled = False

    while not os.path.exists(stop_file):
        t = time.time()
        q = subprocess.run(
            ["nvidia-smi", f"--id={uuid}",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        try:
            util, mem = (int(x) for x in q.split(","))
        except ValueError:
            time.sleep(POLL_S)
            continue
        if baseline_mb is None:
            baseline_mb = mem
        peak_mb = max(peak_mb, mem)
        active = util >= ACTIVE_UTIL or mem > baseline_mb + ACTIVE_MEM_MB
        hi_util_streak = hi_util_streak + 1 if util >= ARM_UTIL else 0
        if not armed and (hi_util_streak >= ARM_STREAK
                          or mem > baseline_mb + ARM_MEM_MB):
            armed = True
            out.write(json.dumps({"t": round(t, 3), "event": "armed",
                                  "util": util, "mem_mb": mem}) + "\n")
        if active:
            if last_active is not None:
                max_gap = max(max_gap, t - last_active)
            last_active = t
        rec = {"t": round(t, 3), "util": util, "mem_mb": mem}
        # [Exp_134] ★재무장 — 원본은 `signaled` 를 한 번 세우면 되돌리지 않아
        #   **첫 작업 하나만 감지하고 그 뒤로는 영원히 침묵**한다. Exp_19 는 작업
        #   하나를 지켜보는 단발 모니터였으므로 그것으로 충분했으나, 상주 데몬으로
        #   띄우면(Exp_134 3-B) 두 번째 작업부터 감지가 없다 — 실측: 10회 중 1회만
        #   이벤트 발화. 새 활동이 무장 조건을 다시 만족하면 재무장한다.
        #   ⚠️ 판정 규칙·임계값·폴링 주기는 **그대로**다(Exp_19 검증치). 바뀐 것은
        #     생애주기뿐이다.
        if signaled and (hi_util_streak >= ARM_STREAK
                         or mem > baseline_mb + ARM_MEM_MB):
            signaled = False
            armed = True
            peak_mb = mem
            max_gap = 0.0
            last_active = t
            out.write(json.dumps({"t": round(t, 3), "event": "rearmed",
                                  "util": util, "mem_mb": mem}) + "\n")
        if armed and not signaled:
            vram_released = (peak_mb - baseline_mb > ARM_MEM_MB and
                             mem < baseline_mb + RELEASE_FRAC * (peak_mb - baseline_mb))
            hold = max(HOLD_MIN_S, HOLD_GAP_K * max_gap)
            idle_exceeded = last_active is not None and (t - last_active) > hold
            if vram_released or idle_exceeded:
                signaled = True
                term = {
                    "t_signal": round(t, 3),
                    "basis": "vram_release" if vram_released else "idle_hold",
                    "confidence": "HIGH" if vram_released else "LOW",
                    "hold_s": round(hold, 2), "peak_mb": peak_mb,
                    "baseline_mb": baseline_mb}
                rec["terminating"] = term
                if events_path:
                    emit_term_event(events_path, uuid,
                                    evidence={**term, "util": util,
                                              "mem_mb": mem,
                                              "periodicity": None})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        time.sleep(max(0.0, POLL_S - (time.time() - t)))
    out.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    events_path = None
    if "--events" in argv:
        i = argv.index("--events")
        events_path = argv[i + 1]
        del argv[i:i + 2]
    uuid, out_path, stop_file = argv[0], argv[1], argv[2]
    monitor(uuid, out_path, stop_file, events_path)


if __name__ == "__main__":
    main()
