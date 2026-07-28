# pattern_matcher.py
# 1단계: 커널 이름 패턴 기반 즉시 killer 식별
# 프로파일링 없이 워크로드 시작 즉시 적용 가능

import re
from dataclasses import dataclass, field
from typing import List

# DL 프레임워크의 killer 커널 패턴 (cuBLAS, cuDNN, FlashAttention 등)
KILLER_PATTERNS = [
    # GEMM 계열
    r".*sgemm.*",
    r".*dgemm.*",
    r".*hgemm.*",
    r".*gemm.*",
    r".*cutlass.*gemm.*",
    # Convolution 계열
    r".*fprop.*",
    r".*xmma.*implicit_gemm.*",
    r".*xmma.*conv.*",
    r".*cudnn.*conv.*",
    # Attention 계열
    r".*fmha.*",
    r".*flash_attn.*",
    r".*attention.*",
    # 기타 대형 행렬 연산
    r".*wgrad.*",
    r".*dgrad.*",
    r".*volta_.*nn.*",
    r".*ampere_.*nn.*",
    r".*sm\d+_xmma.*",
]

# 컴파일된 패턴 (성능)
_COMPILED = [re.compile(p, re.IGNORECASE) for p in KILLER_PATTERNS]

# 워크로드 타입 분류 패턴 (Arithmetic Intensity 기반)
COMPUTE_BOUND_PATTERNS = [r".*gemm.*", r".*fprop.*", r".*xmma.*", r".*cutlass.*"]
MEMORY_BOUND_PATTERNS  = [r".*memcpy.*", r".*elementwise.*", r".*reduce.*",
                           r".*norm.*", r".*softmax.*"]

_COMPUTE_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMPUTE_BOUND_PATTERNS]
_MEMORY_COMPILED  = [re.compile(p, re.IGNORECASE) for p in MEMORY_BOUND_PATTERNS]


@dataclass
class PatternMatchResult:
    killer_names:    List[str]       # 패턴으로 식별된 killer 커널 이름들
    killer_indices:  List[int]       # 시퀀스 내 인덱스 (알 경우)
    workload_type:   str             # COMPUTE_BOUND / MEMORY_BOUND / MIXED
    confidence:      str = "LOW"     # 패턴 매칭은 항상 LOW


def is_killer(kernel_name: str) -> bool:
    return any(p.match(kernel_name) for p in _COMPILED)


def classify_workload(kernel_names: List[str]) -> str:
    compute_count = sum(
        1 for n in kernel_names
        if any(p.match(n) for p in _COMPUTE_COMPILED)
    )
    memory_count = sum(
        1 for n in kernel_names
        if any(p.match(n) for p in _MEMORY_COMPILED)
    )
    total = max(len(kernel_names), 1)

    compute_ratio = compute_count / total
    memory_ratio  = memory_count  / total

    if compute_ratio > 0.4:
        return "COMPUTE_BOUND"
    elif memory_ratio > 0.4:
        return "MEMORY_BOUND"
    else:
        return "MIXED"


def match(kernel_names: List[str]) -> PatternMatchResult:
    """
    커널 이름 목록에서 killer 후보를 즉시 식별한다.
    프로파일링 없이 워크로드 시작 시 즉시 호출 가능.
    """
    killers = []
    indices = []
    for i, name in enumerate(kernel_names):
        if is_killer(name):
            if name not in killers:
                killers.append(name)
            indices.append(i)

    return PatternMatchResult(
        killer_names   = killers,
        killer_indices = indices,
        workload_type  = classify_workload(kernel_names),
        confidence     = "LOW",
    )
