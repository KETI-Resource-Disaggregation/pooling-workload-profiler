"""Phase 0 — K8s Pod annotation 스키마 정의 + 파서.

네임스페이스는 기존 device-plugin의 Extended Resource(keti.re.kr/lsu,
keti.re.kr/gpu-slice)와 동일한 keti.re.kr/ 를 사용한다.

파서 인터페이스 분리 원칙:
  annotation dict → WorkloadSpec (중립 표현) → estimator
  향후 별도 워크로드 디스크립터(YAML 등)는 WorkloadSpec 을 생산하는
  다른 parse_* 함수로 추가하면 estimator 는 무변경.

필수/기본값 정책 (Exp_12_A.md Phase 0-2):
  - model 또는 params 중 하나는 필수 → 둘 다 없으면 거부(SpecError).
    (메모리 추정의 기저인 파라미터 수를 알 방법이 없음)
  - workload-type 필수 → 없으면 거부. (training/inference 는 메모리 식이
    수 배 다르므로 보수적 기본값이 존재하지 않음)
  - dtype 기본 fp32 (보수적 = 가장 큰 dtype → 과대추정 쪽으로 안전)
  - batch 기본 1, tp/pp/dp 기본 1
  - seqlen 기본: 모델 DB의 max_seqlen, DB에 없으면 2048 (근거: 최근
    decoder LLM 의 일반적 학습 context 하한 — LLaMA-1 2048)
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

ANNOTATION_PREFIX = "keti.re.kr/"

# annotation key (prefix 제외) 정의: (필수 여부, 기본값 설명)
ANNOTATION_KEYS = {
    "model":            ("model 또는 params 중 1개 필수", "모델명 (model_db lookup)"),
    "params":           ("model 또는 params 중 1개 필수", "파라미터 수 (예: 6.7B, 124M, 정수)"),
    "dtype":            ("optional, 기본 fp32(보수적)", "fp32/fp16/bf16/int8/fp8/int4"),
    "tp":               ("optional, 기본 1", "tensor parallel degree"),
    "pp":               ("optional, 기본 1", "pipeline parallel degree"),
    "dp":               ("optional, 기본 1", "data parallel degree"),
    "batch":            ("optional, 기본 1", "(마이크로)배치 크기"),
    "seqlen":           ("optional, 기본 model max_seqlen 또는 2048", "시퀀스 길이"),
    "workload-type":    ("필수", "training/inference/batch"),
    "priority":         ("optional", "urgent/normal/bg (없으면 workload-type 로 유도)"),
    "target-slo":       ("optional", "목표 SLO (예: 100ms) — 현재는 priority 승격에만 사용"),
    "logical-unit-id":  ("optional, 기본 model 명", "controller tenant_id 로 쓸 ID (15자 제한)"),
}

WORKLOAD_TYPES_IN = {"training", "inference", "batch"}
PRIORITIES_IN = {"urgent", "normal", "bg"}

DTYPE_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
    "int8": 1.0, "fp8": 1.0, "int4": 0.5,
}

DEFAULT_SEQLEN = 2048


class SpecError(ValueError):
    """필수 필드 누락/값 오류 — 정책상 보수적 추정이 불가능해 거부하는 경우."""


@dataclass
class WorkloadSpec:
    """annotation 과 무관한 중립 워크로드 표현 (파서 인터페이스 경계)."""
    logical_unit_id: str
    workload_type: str                 # training | inference | batch
    model: Optional[str] = None
    params: Optional[int] = None       # 파라미터 개수 (개)
    dtype: str = "fp32"
    tp: int = 1
    pp: int = 1
    dp: int = 1
    batch: int = 1
    seqlen: Optional[int] = None       # None → 모델 DB / DEFAULT_SEQLEN
    priority: Optional[str] = None     # urgent | normal | bg
    target_slo: Optional[str] = None
    defaults_used: list = field(default_factory=list)  # confidence 판정용

    def to_dict(self):
        return asdict(self)


def parse_param_count(text) -> int:
    """'6.7B', '124M', '117k', '124000000' → 정수 파라미터 수."""
    s = str(text).strip().lower().replace(",", "")
    mult = 1
    if s.endswith("b"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("k"):
        mult, s = 1_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        raise SpecError(f"params 값을 해석할 수 없음: {text!r}")


def _get(ann: dict, key: str):
    """prefix 유/무 모두 허용해 annotation 값을 꺼낸다."""
    return ann.get(ANNOTATION_PREFIX + key, ann.get(key))


def _int_field(ann, key, default, defaults_used, minimum=1):
    raw = _get(ann, key)
    if raw is None:
        defaults_used.append(key)
        return default
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise SpecError(f"{key} 는 정수여야 함: {raw!r}")
    if v < minimum:
        raise SpecError(f"{key} 는 {minimum} 이상이어야 함: {v}")
    return v


def parse_annotations(annotations: dict) -> WorkloadSpec:
    """K8s Pod annotation dict → WorkloadSpec. 정책 위반 시 SpecError."""
    defaults_used = []

    model = _get(annotations, "model")
    model = str(model).strip() if model is not None else None

    params_raw = _get(annotations, "params")
    params = parse_param_count(params_raw) if params_raw is not None else None

    if model is None and params is None:
        raise SpecError("거부: model 또는 params 중 하나는 필수 (파라미터 수 추정 불가)")

    wt = _get(annotations, "workload-type")
    if wt is None:
        raise SpecError("거부: workload-type 필수 (training/inference 메모리 식이 상이해 "
                        "보수적 기본값이 존재하지 않음)")
    wt = str(wt).strip().lower()
    if wt not in WORKLOAD_TYPES_IN:
        raise SpecError(f"workload-type 은 {sorted(WORKLOAD_TYPES_IN)} 중 하나: {wt!r}")

    dtype_raw = _get(annotations, "dtype")
    if dtype_raw is None:
        dtype = "fp32"
        defaults_used.append("dtype")
    else:
        dtype = str(dtype_raw).strip().lower()
        if dtype not in DTYPE_BYTES:
            raise SpecError(f"dtype 은 {sorted(DTYPE_BYTES)} 중 하나: {dtype!r}")

    prio_raw = _get(annotations, "priority")
    priority = None
    if prio_raw is not None:
        priority = str(prio_raw).strip().lower()
        if priority not in PRIORITIES_IN:
            raise SpecError(f"priority 는 {sorted(PRIORITIES_IN)} 중 하나: {priority!r}")

    slo = _get(annotations, "target-slo")

    luid_raw = _get(annotations, "logical-unit-id")
    if luid_raw is not None:
        luid = str(luid_raw).strip()
    elif model is not None:
        luid = model
        defaults_used.append("logical-unit-id")
    else:
        luid = f"est-{params // 1_000_000}m"
        defaults_used.append("logical-unit-id")

    seq_raw = _get(annotations, "seqlen")
    seqlen = None
    if seq_raw is not None:
        seqlen = int(str(seq_raw).strip())
        if seqlen < 1:
            raise SpecError(f"seqlen 은 1 이상: {seqlen}")
    # seqlen=None 은 estimator 에서 모델 DB max_seqlen / DEFAULT_SEQLEN 으로 해소

    return WorkloadSpec(
        logical_unit_id=luid,
        workload_type=wt,
        model=model,
        params=params,
        dtype=dtype,
        tp=_int_field(annotations, "tp", 1, defaults_used),
        pp=_int_field(annotations, "pp", 1, defaults_used),
        dp=_int_field(annotations, "dp", 1, defaults_used),
        batch=_int_field(annotations, "batch", 1, defaults_used),
        seqlen=seqlen,
        priority=priority,
        target_slo=str(slo).strip() if slo is not None else None,
        defaults_used=defaults_used,
    )
