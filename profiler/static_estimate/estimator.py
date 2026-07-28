"""추정 엔진 오케스트레이션: WorkloadSpec → controller 출력 JSON.

파이프라인:
  parse_annotations() → resolve_arch() → 메모리 식 → 병렬 분할
  → derive_initial_resources() → 출력 스키마 조립(validate_output 통과 보장)

CLI:
  python3 -m static_estimate.estimator --annotations ann.json
  (ann.json = K8s Pod metadata.annotations dict)
"""
import argparse
import json
import math
import sys

from .schema import WorkloadSpec, parse_annotations, DEFAULT_SEQLEN
from . import model_db
from . import formulas
from .formulas import MB, MemEstimate
from .output_schema import OUTPUT_SCHEMA_VERSION, validate_output, to_json

# annotation workload-type → 커널 특성 분류 (기존 policy_schema 라벨 재사용)
# 근거: PROJECT_CONTEXT §5.3 Agent AI 태깅 — training=Compute·주기적,
#       inference(decode 지배 가정)=Memory·버스트, batch=best-effort(혼합 가정)
WORKLOAD_TYPE_MAP = {
    "training": "COMPUTE_BOUND",
    "inference": "MEMORY_BOUND",
    "batch": "MIXED",
}

# annotation priority → controller int (scheduler.py 관례 0=LOW,1=MED,2=HIGH)
PRIORITY_MAP = {"urgent": 2, "normal": 1, "bg": 0}


def resolve_arch(spec: WorkloadSpec):
    """모델 아키텍처 해소. 반환: (arch|None, inverted: bool)."""
    arch = model_db.lookup(spec.model) if spec.model else None
    if arch is not None:
        return arch, False
    if spec.params is not None:
        return model_db.invert_from_params(spec.params), True
    return None, False


def _default_priority(spec: WorkloadSpec) -> int:
    if spec.priority is not None:
        return PRIORITY_MAP[spec.priority]
    if spec.workload_type == "batch":
        return PRIORITY_MAP["bg"]
    if spec.workload_type == "inference" and spec.target_slo:
        # SLO 가 명시된 추론은 지연 민감 → HIGH (Track 2-2 Urgent 대응 지점)
        return PRIORITY_MAP["urgent"]
    return PRIORITY_MAP["normal"]


def estimate_memory(spec: WorkloadSpec, arch, inverted: bool) -> MemEstimate:
    m = MemEstimate()
    params = arch.params if arch else spec.params

    # spec.params 가 명시되면 DB 값보다 annotation 우선 (사용자가 변형 모델 지정 가능)
    if spec.params is not None:
        params = spec.params

    m.weight_bytes = formulas.split_per_device(
        formulas.weight_memory_bytes(params, spec.dtype), spec.tp, spec.pp)
    m.assumptions.append(f"weights = P×b = {params:,}×{spec.dtype} /(tp·pp)")

    is_transformer = arch is not None and arch.family.startswith("transformer")
    is_decoder = arch is not None and arch.family == "transformer-decoder"
    seqlen = spec.seqlen or (arch.max_seqlen if arch and arch.max_seqlen else DEFAULT_SEQLEN)
    batch_dev = max(1, math.ceil(spec.batch / spec.dp))
    if spec.dp > 1:
        m.assumptions.append(f"batch 는 global 로 해석, 디바이스당 B=ceil({spec.batch}/{spec.dp})={batch_dev}")

    if inverted:
        m.assumptions.append(arch.source)

    # KV 캐시 — decoder LLM 추론에만 해당
    if spec.workload_type in ("inference", "batch"):
        if is_decoder and arch.layers and arch.hidden and arch.heads:
            kv_heads = arch.kv_heads or arch.heads
            m.kv_cache_bytes = formulas.split_per_device(
                formulas.kv_cache_bytes(batch_dev, seqlen, arch.layers, arch.hidden,
                                        arch.heads, kv_heads, spec.dtype),
                spec.tp, spec.pp)
            m.assumptions.append(f"KV = 2·B·S·L·H_kv·b (B={batch_dev}, S={seqlen}) /(tp·pp)")
        elif is_transformer:
            m.kv_cache_bytes = 0.0
            m.assumptions.append("encoder 모델 — autoregressive decode 없음, KV=0")
        elif arch is not None and arch.family == "cnn":
            m.kv_cache_bytes = 0.0
            m.assumptions.append("CNN — KV 캐시 해당 없음, KV=0")
        else:
            m.unestimated["kv_cache"] = "아키텍처(L,H,heads) 불명 — 식 적용 불가"
        # 추론 활성값: CNN 은 실측 테이블(Exp_12-B §5-7)로 보완, 그 외는 식 부재 → 미추정
        # (Exp_12-B 실측: decoder LLM 은 weights+KV 대비 ~1.6% 로 소규모 — 미추정 타당)
        act = None
        if arch is not None and arch.family == "cnn":
            act = formulas.cnn_activation_bytes(arch.name, "inference",
                                                spec.dtype, batch_dev)
        if act is not None:
            m.activation_bytes = act
            m.assumptions.append(
                f"activation = 실측 테이블 (Exp_12-B, 224px) × B={batch_dev}")
        else:
            m.unestimated["activation"] = (
                "추론 활성값 버퍼는 근거 식 부재 (동적 분석 Track 4-2 이관)")

    # 학습 — 모델 상태 + 활성값
    if spec.workload_type == "training":
        m.optimizer_state_bytes = formulas.split_per_device(
            formulas.optimizer_state_bytes(params), spec.tp, spec.pp)
        # optimizer_state_bytes(=P×16) 가 weight 를 포함하므로 중복 계상 방지
        m.weight_bytes = None
        m.assumptions.append("학습 모델 상태 = P×16B (ZeRO: w+g+Adam, weight 포함) /(tp·pp)")
        if is_transformer and arch.layers and arch.hidden and arch.heads:
            m.activation_bytes = formulas.split_per_device(
                formulas.activation_bytes_training(batch_dev, seqlen, arch.layers,
                                                   arch.hidden, arch.heads, spec.dtype),
                spec.tp, spec.pp)
            m.assumptions.append(
                f"activation = L·S·B·H·(34+5aS/H)·(b/2) (Korthikanti 2022, "
                f"recomputation 없음 가정, B={batch_dev}, S={seqlen}) /(tp·pp)")
        else:
            act = None
            if arch is not None and arch.family == "cnn":
                act = formulas.cnn_activation_bytes(arch.name, "training",
                                                    spec.dtype, batch_dev)
            if act is not None:
                m.activation_bytes = act
                m.assumptions.append(
                    f"activation = 실측 테이블 (Exp_12-B, 224px) × B={batch_dev} — "
                    f"배치 선형성 실측 확인")
            else:
                m.unestimated["activation"] = (
                    "transformer 외 모델의 학습 활성값 식 부재 "
                    "(실측 테이블 외 조합 — 외삽하지 않음)")

    m.assumptions.append(
        f"CUDA context overhead {formulas.CUDA_CONTEXT_OVERHEAD_MB}MB "
        f"(Exp_12-B 실측값 — 장비/드라이버 의존, 스택 변경 시 재실측)")
    return m


def estimate(annotations: dict) -> dict:
    """K8s Pod annotation dict → controller 출력 JSON (dict).

    반환값은 validate_output() 을 통과하는 정적 초기 추정값.
    SpecError 는 호출자에게 전파 (정책상 거부).
    """
    spec = parse_annotations(annotations)
    arch, inverted = resolve_arch(spec)
    mem = estimate_memory(spec, arch, inverted)
    est_mem_mb = mem.total_mb()
    res = formulas.derive_initial_resources(est_mem_mb)

    if res["exceeds_device"]:
        mem.unestimated["fit"] = (
            f"추정 메모리 {est_mem_mb}MB 가 단일 디바이스({res['device']} "
            f"{formulas.DEVICE_PROFILE['mem_mb']}MB) 초과 — 병렬 degree 재검토 필요")

    # confidence 판정 (policy_schema 의 LOW/HIGH 관례):
    #   HIGH = DB 아키텍처 + dtype 명시 + 주요 항목 미추정 없음
    #   LOW  = 역산/미등록 모델, dtype 기본값 사용, 주요 미추정 존재
    # 예외: 추론의 activation 미추정은 KV 외 transient buffer 로 weight/KV 대비
    #       소규모라 강등 요인에서 제외. Exp_12-B 실측: decoder LLM(opt-6.7b) 222MB
    #       (총량의 1.6%) — 타당. 단 소형 encoder(bert-base b=32)는 281MB 로
    #       weights(208MB)와 동급이나 절대량이 작아 예외 유지 (경계는 Track 4-2)
    critical_unest = {k: v for k, v in mem.unestimated.items()
                      if not (k == "activation"
                              and spec.workload_type in ("inference", "batch"))}
    high = (arch is not None and not inverted
            and "dtype" not in spec.defaults_used
            and not critical_unest)
    confidence = "HIGH" if high else "LOW"

    def _mb(x):
        return None if x is None else math.ceil(x / MB)

    tenant_id = spec.logical_unit_id[:15]   # registry.py:46 절단 한계 선반영

    out = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source": "static_estimate",
        "logical_unit_id": spec.logical_unit_id,
        "est_mem_mb": est_mem_mb,
        "sm_init": res["sm_init"],
        "quota_mb": res["quota_mb"],
        "token_budget_us": res["token_budget_us"],
        "lsu_est": float(res["lsu_est"]),
        "workload_type": WORKLOAD_TYPE_MAP[spec.workload_type],
        "priority": _default_priority(spec),
        "confidence": confidence,
        "breakdown_mb": {
            "weights": _mb(mem.weight_bytes),
            "kv_cache": _mb(mem.kv_cache_bytes),
            "activation": _mb(mem.activation_bytes),
            "optimizer_state": _mb(mem.optimizer_state_bytes),
            "context_overhead": _mb(mem.context_overhead_bytes),
        },
        "unestimated": mem.unestimated,
        "assumptions": mem.assumptions,
        "register_mapping": {   # 연결 지점 문서화 — 실제 호출은 하지 않음
            "tenant_id": tenant_id,
            "virtual_sm": res["sm_init"],
            "virtual_mem_mb": est_mem_mb,
        },
        "note": "초기 추정값(정적) — Exp_12-B 실측 보정 대상. 확정값 아님.",
    }
    errs = validate_output(out)
    if errs:   # 내부 불변식 — 스키마와 조립 코드가 어긋나면 즉시 드러낸다
        raise AssertionError(f"output schema violation: {errs}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="정적 프로파일링 추정 엔진 (Exp_12-A)")
    ap.add_argument("--annotations", required=True,
                    help="K8s Pod metadata.annotations JSON 파일 경로 (- 는 stdin)")
    args = ap.parse_args(argv)
    raw = sys.stdin.read() if args.annotations == "-" else open(args.annotations).read()
    ann = json.loads(raw)
    # Pod manifest 전체가 와도 annotations 만 추출
    if "metadata" in ann and isinstance(ann["metadata"], dict):
        ann = ann["metadata"].get("annotations", {})
    print(to_json(estimate(ann)))


if __name__ == "__main__":
    main()
