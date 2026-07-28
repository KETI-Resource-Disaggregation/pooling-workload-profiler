"""Phase 2 — controller 출력 인터페이스 정의.

기존 controller API 확인 결과 (Exp_12-A 탐사):
  - 등록 진입점은 prism_controller.py `POST /register`
    {tenant_id, virtual_sm, virtual_mem_mb, weight} 뿐이며
    quota/token_budget/confidence 개념은 controller/SHM 에 존재하지 않음.
  → 따라서 본 모듈이 JSON 출력 스키마를 신규 정의하고, /register 로의
    필드 매핑(연결 지점)만 문서화한다. ★ 실제 register 호출은 하지 않음.

/register 매핑 (소비 측 연결 지점):
  logical_unit_id → tenant_id       (registry.py:46 — 15자 초과분 절단됨)
  est_mem_mb      → virtual_mem_mb  (int, MB)
  sm_init         → virtual_sm      (int, 물리 SM 개수 단위; mps_pct 는 controller 가 유도)
  priority        → POST /policy/priority {tenant_idx, priority}
                    (controller 관례: 0=LOW, 1=MED, 2=HIGH — scheduler.py:27-29)
  quota_mb / token_budget_us / lsu_est / workload_type / confidence
                  → 현 controller 미소비. 소비 지점 후보:
                    quota_mb → libbless mem_quota 명령, token_budget_us → Track 1-1
                    토큰 예산, lsu_est → device-plugin keti.re.kr/lsu 리소스.
"""
import json

OUTPUT_SCHEMA_VERSION = 1

# 필수 필드: (이름, 타입, 허용값 검사 함수 또는 None)
_PRIORITY_VALUES = {0, 1, 2}                      # controller 관례 LOW/MED/HIGH
_WORKLOAD_TYPES_OUT = {"COMPUTE_BOUND", "MEMORY_BOUND", "MIXED", "UNKNOWN"}
_CONFIDENCE = {"LOW", "HIGH"}                     # profiler/policy_schema.py 관례 재사용

REQUIRED_FIELDS = {
    "schema_version": int,
    "source": str,            # 항상 "static_estimate"
    "logical_unit_id": str,
    "est_mem_mb": int,
    "sm_init": int,
    "quota_mb": int,
    "token_budget_us": int,
    "lsu_est": float,
    "workload_type": str,
    "priority": int,
    "confidence": str,
    "breakdown_mb": dict,     # {weights|kv_cache|activation|optimizer_state|context_overhead: MB|None}
    "unestimated": dict,      # {항목: 미추정 사유}
    "assumptions": list,      # 사용한 가정/식 요약 문자열 목록
    "register_mapping": dict, # /register 로 넘길 body (연결 지점 문서화용 — 미호출)
    "note": str,
}


def validate_output(out: dict) -> list:
    """스키마 위반 목록 반환 (빈 리스트 = 통과). policy_schema.validate 관례."""
    errs = []
    for k, t in REQUIRED_FIELDS.items():
        if k not in out:
            errs.append(f"missing field: {k}")
            continue
        v = out[k]
        if t is float:
            if not isinstance(v, (int, float)):
                errs.append(f"{k}: expected number, got {type(v).__name__}")
        elif not isinstance(v, t):
            errs.append(f"{k}: expected {t.__name__}, got {type(v).__name__}")
    if not errs:
        if out["source"] != "static_estimate":
            errs.append("source must be 'static_estimate'")
        if out["priority"] not in _PRIORITY_VALUES:
            errs.append(f"priority must be one of {sorted(_PRIORITY_VALUES)}")
        if out["workload_type"] not in _WORKLOAD_TYPES_OUT:
            errs.append(f"workload_type must be one of {sorted(_WORKLOAD_TYPES_OUT)}")
        if out["confidence"] not in _CONFIDENCE:
            errs.append(f"confidence must be one of {sorted(_CONFIDENCE)}")
        if len(out["register_mapping"].get("tenant_id", "")) > 15:
            errs.append("register_mapping.tenant_id exceeds 15 chars (registry.py 절단 한계)")
    return errs


def to_json(out: dict) -> str:
    return json.dumps(out, ensure_ascii=False, indent=2)
