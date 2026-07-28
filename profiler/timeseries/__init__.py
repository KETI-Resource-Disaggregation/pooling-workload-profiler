"""Trace Analyzer 4-3(주기성 검출) + 4-4(종료 임박 감지) — 시계열 엔진 (Exp_14).

입력:
  4-3 — 커널 실행 시계열 (Exp_13/14 kernel_tracer 트레이스의 start_ns)
  4-4 — 활용률 시계열 (nvidia-smi/DCGM, 100ms 급) + 4-3 의 주기 정보
4-4 는 4-3 의 주기를 알아야 "iteration 간 휴지"와 "진짜 종료"를 구분한다.
"""
from .periodicity import (bin_events, detect_period, detect_period_scan,
                          mem_peak_analysis)
from .termination import detect_termination, hold_from_gaps
from .schema import build_output, validate_output

__all__ = ["bin_events", "detect_period", "detect_period_scan",
           "mem_peak_analysis", "detect_termination", "hold_from_gaps",
           "build_output", "validate_output"]
