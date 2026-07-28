"""Trace Analyzer 4-2 — 커널 분류 엔진 (동적 관측, Exp_13).

CUPTI Activity 트레이스(커널명·GPU 실행시간·grid/block)를 입력으로
상위 커널 식별 + Compute/Memory 분류를 산출한다.
4-1(static_estimate)이 이관한 항목(SLO 기반 SM 요구, 추론 활성값,
prefill/decode 구분)의 동적 확인을 담당.
"""
from .patterns import classify_kernel_name
from .classify import parse_trace, aggregate, summarize, validate_output

__all__ = ["classify_kernel_name", "parse_trace", "aggregate",
           "summarize", "validate_output"]
