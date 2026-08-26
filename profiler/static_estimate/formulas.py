"""Phase 1 — 메모리/자원 추정 계산식. 모든 식에 근거를 명시한다.

★ 원칙: 식으로 유도 불가한 항목은 값을 만들지 않고 미추정(None + 사유)으로
  돌려준다. 모든 결과는 초기 추정값이며 Exp_12-B 실측 보정 대상.

기호: P=파라미터 수, b=dtype 바이트, B=배치, S=시퀀스 길이,
      L=레이어 수, H=hidden, a=attention heads, tp/pp/dp=병렬 degree
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from .schema import DTYPE_BYTES

MB = 1024 * 1024
GB = 1024 * MB

# ─────────────────────────────────────────────────────────────────────
# 잠정 상수 (전부 Exp_12-B 실측 보정 대상)
# ─────────────────────────────────────────────────────────────────────

# CUDA context + cuBLAS/cuDNN 핸들·workspace 의 프로세스당 고정 오버헤드.
# 근거: Exp_12-B 실측 (2026-07-03, 노드 230 A6000, driver 575.64.03,
# torch 2.9.0+cu128) — 프로세스 7개에서 nvidia-smi 프로세스 메모리 −
# max_reserved = 314~330MB 로 일관. 관측 상한 330 채택 (장비/드라이버 의존값,
# 다른 스택에서는 재실측 필요).
CUDA_CONTEXT_OVERHEAD_MB = 330          # measured — Exp_12B_estimate_calibration_230

# Adam(mixed precision) 학습 시 파라미터당 모델 상태 바이트.
# 근거: Rajbhandari et al. 2020 (ZeRO) — fp16 weight(2) + fp16 grad(2)
#       + fp32 master(4) + Adam m(4) + v(4) = 16 bytes/param.
#       fp32 학습도 w(4)+g(4)+m(4)+v(4) = 16 으로 동일.
ADAM_BYTES_PER_PARAM = 16

# CNN 활성값 실측 테이블 (MB/sample) — 식 부재 항목의 실측 보완 (Exp_12-B §5-7).
# 근거: Exp_12-B 실측 (노드 230 A6000, 입력 3×224×224):
#   resnet50 train fp32: b=32→2601MB, b=64→5255MB (81.3/82.1 MB/sample —
#     배치 선형성 b64/b32=2.02 로 확인) → 82.1 채택 (상한)
#   resnet50 infer fp16: b=64→373MB → 5.83 MB/sample
# activation = max_alloc − weights − grads − opt_states 로 산출.
# ★ 실측된 (모델, 모드, dtype) 조합만 수록 — 그 외 조합은 외삽하지 않고 미추정 유지.
CNN_ACTIVATION_MB_PER_SAMPLE = {
    ("resnet50", "training", "fp32"): 82.1,
    ("resnet50", "inference", "fp16"): 5.83,
}


def cnn_activation_bytes(model_name: str, mode: str, dtype: str,
                         batch: int) -> Optional[float]:
    """CNN 활성값 = 실측 per-sample × B (배치 선형성 실측 확인, 224px 입력 기준).

    실측 테이블에 없는 (모델, 모드, dtype) 조합은 None (미추정 유지).
    mode: "training" | "inference" (batch 추론은 inference 로 취급).
    """
    per_sample = CNN_ACTIVATION_MB_PER_SAMPLE.get((model_name, mode, dtype))
    if per_sample is None:
        return None
    return per_sample * batch * MB


# quota 여유 마진 (Exp_12-B §5-6) — allocator fragmentation(reserved−allocated)
# 실측 상한 포락선. 근거: 워크로드 6개 실측 141~753MB, 비율은 소형일수록 커짐
# (bert 30% ~ opt-6.7b 2%). margin = clamp(0.15×est, 256, 1024) 가 전 관측 포괄.
# ★ n=6 포락선 — provisional. allocator 정책(torch caching) 의존.
QUOTA_MARGIN_RATIO = 0.15
QUOTA_MARGIN_MIN_MB = 256
QUOTA_MARGIN_MAX_MB = 1024


def quota_margin_mb(est_mem_mb: int) -> int:
    """quota 여유 마진(MB) = min(1024, max(256, 0.15×est)) — 실측 포락선."""
    return int(min(QUOTA_MARGIN_MAX_MB,
                   max(QUOTA_MARGIN_MIN_MB, QUOTA_MARGIN_RATIO * est_mem_mb)))


# 노드 230 실장비 프로파일 (kraken_shm.py:214 — A6000 SM 84 / lsu catalog 실측 65)
DEVICE_PROFILE = {
    "name": "NVIDIA RTX A6000",
    "mem_mb": 48 * 1024,     # 48 GiB
    "sm_total": 84,          # shm/kraken_shm.py:214 (compute capability 8.6)
    "lsu": 65,               # lsu/catalog.json 2026-06-18 실측 (provisional)
}

# controller round 기본값 (shm/kraken_shm.py:124 — 100ms)
ROUND_DURATION_DEFAULT_US = 100_000


@dataclass
class MemEstimate:
    """디바이스(논리 유닛) 1개당 메모리 추정 내역. bytes 단위, None=미추정."""
    weight_bytes: Optional[float] = None
    kv_cache_bytes: Optional[float] = None
    activation_bytes: Optional[float] = None
    optimizer_state_bytes: Optional[float] = None   # weight 포함한 모델 상태 전체
    context_overhead_bytes: float = CUDA_CONTEXT_OVERHEAD_MB * MB
    unestimated: dict = field(default_factory=dict)  # {항목: 사유}
    assumptions: list = field(default_factory=list)

    def total_bytes(self) -> float:
        parts = [self.weight_bytes, self.kv_cache_bytes,
                 self.activation_bytes, self.optimizer_state_bytes]
        return sum(p for p in parts if p is not None) + self.context_overhead_bytes

    def total_mb(self) -> int:
        return math.ceil(self.total_bytes() / MB)


# ─────────────────────────────────────────────────────────────────────
# 개별 식
# ─────────────────────────────────────────────────────────────────────

def weight_memory_bytes(params: int, dtype: str) -> float:
    """가중치 메모리 = P × b_dtype.  근거: 정의 그대로 (저장 바이트 수)."""
    return params * DTYPE_BYTES[dtype]


def kv_cache_bytes(batch: int, seqlen: int, layers: int, hidden: int,
                   heads: int, kv_heads: int, dtype: str) -> float:
    """KV 캐시 = 2(K,V) × B × S × L × H×(kv_heads/heads) × b_dtype.

    근거: autoregressive decode 시 레이어마다 K,V 텐서 [B,S,H_kv] 를 캐시.
    GQA 면 H_kv = H·(kv_heads/heads). (Pope et al. 2022 등 표준 식)
    """
    h_kv = hidden * (kv_heads / heads)
    return 2.0 * batch * seqlen * layers * h_kv * DTYPE_BYTES[dtype]


def activation_bytes_training(batch: int, seqlen: int, layers: int,
                              hidden: int, heads: int, dtype: str) -> float:
    """학습 활성값 메모리 (recomputation 없음, 병렬 미반영 기준).

    근거: Korthikanti et al. 2022 "Reducing Activation Recomputation in
    Large Transformer Models" — 레이어당 활성값 ≈ S·B·H·(34 + 5·a·S/H) bytes.
    (2바이트 활성값 기준 식이므로 dtype 에 따라 ×(b/2) 스케일 — 근사)
    """
    per_layer = seqlen * batch * hidden * (34.0 + 5.0 * heads * seqlen / hidden)
    scale = DTYPE_BYTES[dtype] / 2.0
    return per_layer * layers * scale


def optimizer_state_bytes(params: int) -> float:
    """학습 모델 상태(weight+grad+Adam) = P × 16.  근거: ZeRO (상수 주석 참조)."""
    return params * ADAM_BYTES_PER_PARAM


# ─────────────────────────────────────────────────────────────────────
# 병렬 분할
# ─────────────────────────────────────────────────────────────────────

def split_per_device(value: Optional[float], tp: int, pp: int) -> Optional[float]:
    """TP는 텐서(heads/hidden 축), PP는 레이어 축으로 분할 → /(tp·pp).

    DP 는 모델 상태를 복제하므로 나누지 않는다 (배치만 분산).
    근거: Megatron-LM (Shoeybi et al. 2019) / GPipe 의 표준 분할 방식.
    한계: TP 시 활성값의 비분할 항(예: LayerNorm 입력)은 미반영 → 근사,
          sequence parallelism 미가정. B 에서 보정.
    """
    if value is None:
        return None
    return value / (tp * pp)


# ─────────────────────────────────────────────────────────────────────
# 초기 자원값 유도 (전부 provisional proxy — B 보정 대상)
# ─────────────────────────────────────────────────────────────────────

def derive_initial_resources(est_mem_mb: int, device: dict = DEVICE_PROFILE) -> dict:
    """예상 메모리 → SM/LSU/Quota/토큰예산 초기값.

    proxy 규칙 (잠정): 자원 footprint 가 메모리 footprint 에 비례한다고 가정.
      fraction    = est_mem / device_mem       (디바이스 점유 비율)
      sm_init     = round(fraction × SM_total)  [1, SM_total] 로 clamp
      lsu_est     = fraction × device_lsu       (LSU catalog 은 디바이스 전체
                    용량만 정의하므로 '디바이스 비율 × LSU' 로 환산 — provisional)
      quota_mb    = est_mem_mb + quota_margin_mb(est)  (마진 = fragmentation
                    실측 포락선, Exp_12-B §5-6. context 는 est_mem 에 이미 포함)
      token_budget_us = round(fraction × round_duration 기본 100ms)
                    (Track 1-3 의 워크로드별 예산 정책은 동적 분석 몫 —
                     정적 단계는 점유 비율 비례 proxy 만 제공)

    ★ Exp_12-B 실측 결과: mem_fraction 0.011~0.32 인 워크로드 5종 전부 정상상태
      GPU util 96~100% — 메모리 비율은 SM "수요"를 예측하지 못한다 (반증 확인).
      따라서 sm_init/lsu_est/token_budget 은 자원 요구 추정이 아니라 멀티테넌트
      초기 배분 시드(디바이스 지분 비례)로만 유효. SLO 기반 산정은 동적 분석
      (Track 4-2) 이관.
    """
    fraction = est_mem_mb / device["mem_mb"]
    over = fraction > 1.0
    f = min(fraction, 1.0)
    return {
        "device": device["name"],
        "mem_fraction": round(fraction, 4),
        "exceeds_device": over,
        "sm_init": max(1, min(device["sm_total"], round(f * device["sm_total"]))),
        "lsu_est": round(f * device["lsu"], 2),
        "quota_mb": est_mem_mb + quota_margin_mb(est_mem_mb),
        "token_budget_us": max(1, round(f * ROUND_DURATION_DEFAULT_US)),
    }
