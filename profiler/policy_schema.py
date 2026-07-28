# policy_schema.py
# killer_policy JSON 포맷의 단일 소스
# profiler(writer)와 runtime(reader) 모두 이 파일을 참조한다

SCHEMA_VERSION = 1

# killer_policy_{fingerprint}.json 구조:
# {
#   "schema_version":   1,
#   "fingerprint":      "a3f9c2d1e8b4",   SHA256 앞 12자
#   "workload_type":    "COMPUTE_BOUND",  COMPUTE_BOUND / MEMORY_BOUND / MIXED
#   "kernels_per_iter": 460,
#   "killer_indices":   [3, 7, 15, 23],   첫 iteration 내 인덱스
#   "killer_names":     ["volta_sgemm_128x64_nn", ...],
#   "killer_ratio":     0.213,            killer / total_kernels
#   "top_pct":          0.20,
#   "avg_killer_us":    8400.5,           killer 평균 실행시간 (μs)
#   "confidence":       "HIGH",           LOW(pattern) / HIGH(solo profiled)
#   "profiled_at":      "2026-03-16T10:00:00"
# }

REQUIRED_FIELDS = [
    "schema_version",
    "fingerprint",
    "workload_type",
    "kernels_per_iter",
    "killer_indices",
    "killer_names",
    "killer_ratio",
    "top_pct",
    "avg_killer_us",
    "confidence",
]

WORKLOAD_TYPES = {"COMPUTE_BOUND", "MEMORY_BOUND", "MIXED"}
CONFIDENCE     = {"LOW", "HIGH"}


def validate(policy: dict) -> bool:
    if not all(f in policy for f in REQUIRED_FIELDS):
        return False
    if policy.get("workload_type") not in WORKLOAD_TYPES:
        return False
    if policy.get("confidence") not in CONFIDENCE:
        return False
    if policy.get("schema_version") != SCHEMA_VERSION:
        return False
    return True


def summary(policy: dict) -> str:
    return (
        f"[{policy.get('confidence','?')}] "
        f"type={policy.get('workload_type','?')} "
        f"killers={len(policy.get('killer_names', []))} "
        f"({policy.get('killer_ratio', 0):.1%}) "
        f"avg={policy.get('avg_killer_us', 0):.0f}μs "
        f"fp={policy.get('fingerprint','?')}"
    )
