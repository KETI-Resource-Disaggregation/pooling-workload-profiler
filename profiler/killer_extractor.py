# killer_extractor.py
# Solo 프로파일링 데이터에서 killer_policy를 추출한다.
# 입력: torch.profiler 또는 CUPTI 수집 결과 (커널별 이름 + 실행시간)
# 출력: killer_policy dict (policy_schema.py 포맷)

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

from policy_schema   import SCHEMA_VERSION, validate
from pattern_matcher import classify_workload
from fingerprint     import compute as compute_fp, save_cache


TOP_PCT_DEFAULT = 0.20   # 상위 20%를 killer로 지정


@dataclass
class KernelRecord:
    """단일 커널 실행 기록"""
    name:         str
    duration_us:  float   # 실행 시간 (μs)
    index:        int     # iteration 내 순서 인덱스


@dataclass
class KillerPolicy:
    schema_version:   int
    fingerprint:      str
    workload_type:    str           # COMPUTE_BOUND / MEMORY_BOUND / MIXED
    kernels_per_iter: int
    killer_indices:   List[int]
    killer_names:     List[str]
    killer_ratio:     float
    top_pct:          float
    avg_killer_us:    float         # killer 커널 평균 실행 시간
    confidence:       str           # LOW (pattern) / HIGH (solo profiled)
    profiled_at:      str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    def to_dict(self) -> dict:
        return asdict(self)


def extract(
    records:    List[KernelRecord],
    top_pct:    float = TOP_PCT_DEFAULT,
) -> KillerPolicy:
    """
    Solo 프로파일링 결과(KernelRecord 리스트)에서 KillerPolicy를 추출한다.

    Args:
        records:  여러 iteration의 커널 기록 (반복 실행 결과)
        top_pct:  실행 시간 상위 N%를 killer로 지정

    Returns:
        KillerPolicy (캐시 자동 저장)
    """
    if not records:
        raise ValueError("records가 비어 있습니다")

    all_names = [r.name for r in records]

    # iteration당 커널 수 추정
    kernels_per_iter = _estimate_kernels_per_iter(all_names)

    # 커널별 평균 실행 시간 계산
    time_by_name: dict[str, list[float]] = {}
    for r in records:
        time_by_name.setdefault(r.name, []).append(r.duration_us)
    avg_by_name = {n: sum(ts) / len(ts) for n, ts in time_by_name.items()}

    # 상위 top_pct killer 선택
    ranked   = sorted(avg_by_name.items(), key=lambda x: x[1], reverse=True)
    n_killer = max(1, int(len(ranked) * top_pct))
    killer_set = {name for name, _ in ranked[:n_killer]}

    # 첫 iteration 기준으로 인덱스/이름 추출
    first_iter     = records[:kernels_per_iter]
    killer_indices = [r.index for r in first_iter if r.name in killer_set]
    killer_names   = [r.name  for r in first_iter if r.name in killer_set]

    avg_killer_us = (
        sum(avg_by_name[n] for n in killer_set if n in avg_by_name)
        / max(len(killer_set), 1)
    )

    fp = compute_fp(all_names)

    policy = KillerPolicy(
        schema_version   = SCHEMA_VERSION,
        fingerprint      = fp,
        workload_type    = classify_workload(all_names),
        kernels_per_iter = kernels_per_iter,
        killer_indices   = killer_indices,
        killer_names     = killer_names,
        killer_ratio     = round(len(killer_indices) / max(kernels_per_iter, 1), 3),
        top_pct          = top_pct,
        avg_killer_us    = round(avg_killer_us, 2),
        confidence       = "HIGH",
    )

    save_cache(policy.to_dict())
    return policy


def _estimate_kernels_per_iter(names: list[str]) -> int:
    """
    커널 이름 시퀀스에서 iteration 경계를 추정한다.
    첫 번째 커널 이름이 다시 등장하는 위치까지의 거리를 iteration 길이로 사용.
    """
    if not names:
        return 1
    first = names[0]
    for i in range(1, len(names)):
        if names[i] == first:
            return i
    return len(names)


def from_torch_profiler(profiler_output, **kwargs) -> KillerPolicy:
    """
    torch.profiler.profile() 결과를 KernelRecord 리스트로 변환 후 추출한다.

    사용 예:
        import torch
        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                model(inputs)

        policy = from_torch_profiler(prof)
    """
    records = []
    for i, event in enumerate(profiler_output.events()):
        # CUDA 커널 이벤트만 수집
        if hasattr(event, 'device_type') and str(event.device_type) == 'DeviceType.CUDA':
            if event.cuda_time > 0:
                records.append(KernelRecord(
                    name        = event.name,
                    duration_us = event.cuda_time,
                    index       = i,
                ))
    return extract(records, **kwargs)


def load(fingerprint_or_model: str, profiles_dir: str = "profiles") -> dict:
    """저장된 killer_policy JSON 로드."""
    base = Path(profiles_dir)
    # fingerprint 기반 시도
    p = base / f"killer_policy_{fingerprint_or_model}.json"
    if not p.exists():
        # 캐시 디렉토리 시도
        p = base / "cache" / f"killer_policy_{fingerprint_or_model}.json"
    with open(p) as f:
        d = json.load(f)
    assert validate(d), f"유효하지 않은 policy 포맷: {p}"
    return d


def save_named(policy: KillerPolicy, name: str,
               profiles_dir: str = "profiles") -> Path:
    """killer_policy를 이름 기반 파일로 추가 저장 (캐시와 별개)."""
    out = Path(profiles_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"killer_policy_{name}.json"
    with open(p, "w") as f:
        json.dump(policy.to_dict(), f, indent=2)
    return p
