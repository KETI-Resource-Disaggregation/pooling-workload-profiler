"""트레이스 파싱 → 상위 커널 랭킹 → Compute/Memory 분류 → 4-2 출력 스키마.

입력: kernel_tracer.so(CUPTI Activity API) 가 남긴 파이프 구분 트레이스
  K|start_ns|dur_ns|gx,gy,gz|bx,by,bz|stream|regs|smem|name
  M|start_ns|dur_ns|bytes|copyKind      S|start_ns|dur_ns|bytes

출력 스키마 (schema_version=1, source="kernel_class"):
  {logical_unit_id, n_kernels, window_ms, top_kernels[],
   compute_memory_ratio, uncertain_time_share,
   classification: COMPUTE_BOUND|MEMORY_BOUND|MIXED|UNKNOWN,
   prefill_decode_hint: prefill|decode|n/a, confidence}

분류 임계 (잠정 — 시간 점유 기반, 근거는 report Phase 2):
  compute_memory_ratio = compute 시간 / (compute+memory 시간)
  ≥0.6 → COMPUTE_BOUND, ≤0.4 → MEMORY_BOUND, 그 외 MIXED.
  UNCERTAIN 시간 점유 >0.2 → confidence LOW (분류 근거 부족을 정직하게 표기).
"""
import argparse
import json
import shutil
import subprocess
from collections import defaultdict

from .patterns import classify_kernel_name

SCHEMA_VERSION = 1

# 잠정 임계값 — 시간 점유 기반 (Exp_13 report Phase 2 에 근거 기록)
COMPUTE_BOUND_MIN_RATIO = 0.6
MEMORY_BOUND_MAX_RATIO = 0.4
UNCERTAIN_LOW_CONF_SHARE = 0.2
# prefill/decode 힌트 (LLM 추론 트레이스에만 의미):
#   decode 시그니처 = gemv/splitkv(MEMORY_OVERRIDE) 시간 점유 ≥0.3
#   prefill 시그니처 = gemm 류 점유 ≥0.5 이면서 decode 시그니처 <0.05
DECODE_SIG_MIN = 0.3
PREFILL_SIG_MIN = 0.5
DECODE_SIG_ABSENT = 0.05


def parse_trace(path):
    """트레이스 파일(.gz 허용) → {"kernels": [...], "memcpy_bytes": int, "memset_bytes": int}."""
    import gzip
    kernels, memcpy_b, memset_b = [], 0, 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("K|"):
                p = line.split("|")
                if len(p) < 9:
                    continue
                gx, gy, gz = (int(x) for x in p[3].split(","))
                kernels.append({
                    "start_ns": int(p[1]), "dur_ns": int(p[2]),
                    "blocks": gx * gy * gz,
                    "stream": int(p[5]), "regs": int(p[6]),
                    "smem": int(p[7]), "name": p[8],
                })
            elif line.startswith("M|"):
                memcpy_b += int(line.split("|")[3])
            elif line.startswith("S|"):
                memset_b += int(line.split("|")[3])
    return {"kernels": kernels, "memcpy_bytes": memcpy_b,
            "memset_bytes": memset_b}


def _demangle(names):
    """c++filt 일괄 demangle (없으면 원문 유지)."""
    if not shutil.which("c++filt"):
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names),
                         capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


def aggregate(kernels):
    """커널명별 집계 → 누적시간 내림차순 리스트."""
    agg = defaultdict(lambda: {"count": 0, "total_ns": 0, "blocks_sum": 0})
    for k in kernels:
        a = agg[k["name"]]
        a["count"] += 1
        a["total_ns"] += k["dur_ns"]
        a["blocks_sum"] += k["blocks"]
    demangled = _demangle(list(agg.keys()))
    rows = []
    for name, a in agg.items():
        rows.append({
            "name": demangled[name],
            "count": a["count"],
            "total_ms": round(a["total_ns"] / 1e6, 2),
            "avg_us": round(a["total_ns"] / a["count"] / 1e3, 1),
            "avg_blocks": round(a["blocks_sum"] / a["count"], 1),
        })
    rows.sort(key=lambda r: -r["total_ms"])
    return rows


def summarize(trace_path, logical_unit_id, top_n=10, sm_total=84,
              workload_type=None):
    """트레이스 → 4-2 출력 dict (schema v1).

    workload_type: "inference"|"training"|"batch"|None.
    prefill/decode 힌트는 LLM 추론 국면 시그니처라 inference 에만 산출 —
    학습 GEMM 지배 트레이스에 'prefill' 을 붙이는 오독을 막는다.
    """
    tr = parse_trace(trace_path)
    kernels = tr["kernels"]
    if not kernels:
        raise ValueError(f"트레이스에 커널 레코드 없음: {trace_path} (sanity gate 실패)")
    rows = aggregate(kernels)

    t = {"COMPUTE": 0.0, "MEMORY": 0.0, "UNCERTAIN": 0.0}
    override_ms = gemm_ms = 0.0
    for r in rows:
        cls, basis = classify_kernel_name(r["name"])
        r["class"], r["basis"] = cls, basis
        t[cls] += r["total_ms"]
        low = r["name"].lower()
        if "gemv" in low or "splitkv" in low:
            override_ms += r["total_ms"]
        elif "gemm" in low:
            gemm_ms += r["total_ms"]

    total_ms = sum(t.values())
    classified = t["COMPUTE"] + t["MEMORY"]
    ratio = (t["COMPUTE"] / classified) if classified > 0 else None
    uncertain_share = t["UNCERTAIN"] / total_ms if total_ms > 0 else 1.0

    if ratio is None:
        classification = "UNKNOWN"
    elif ratio >= COMPUTE_BOUND_MIN_RATIO:
        classification = "COMPUTE_BOUND"
    elif ratio <= MEMORY_BOUND_MAX_RATIO:
        classification = "MEMORY_BOUND"
    else:
        classification = "MIXED"

    decode_sig = override_ms / total_ms if total_ms else 0.0
    prefill_sig = gemm_ms / total_ms if total_ms else 0.0
    hint = "n/a"
    if workload_type in (None, "inference"):   # 추론(또는 불명)에만 의미
        if decode_sig >= DECODE_SIG_MIN:
            hint = "decode"
        elif prefill_sig >= PREFILL_SIG_MIN and decode_sig < DECODE_SIG_ABSENT:
            hint = "prefill"

    # 상위 커널의 grid 크기 → SM 점유 힌트 (SLO 없이 가능한 선까지 — ①)
    top = rows[:top_n]
    for r in top:
        r["sm_fill_hint"] = ("full" if r["avg_blocks"] >= sm_total
                             else f"partial({r['avg_blocks']:.0f}/{sm_total} blocks)")

    out = {
        "schema_version": SCHEMA_VERSION,
        "source": "kernel_class",
        "logical_unit_id": logical_unit_id,
        "n_kernels": len(kernels),
        "n_unique_kernels": len(rows),
        "window_ms": round(total_ms, 1),
        "memcpy_mb": round(tr["memcpy_bytes"] / 2**20, 1),
        "top_kernels": top,
        "compute_memory_ratio": round(ratio, 4) if ratio is not None else None,
        "uncertain_time_share": round(uncertain_share, 4),
        "classification": classification,
        "prefill_decode_hint": hint,
        "confidence": "LOW" if uncertain_share > UNCERTAIN_LOW_CONF_SHARE else "HIGH",
        "note": "동적 관측(kernel_class) — 분류 근거는 커널명 패턴+roofline 논거, "
                "실측 연산강도 계측 아님. 임계값 잠정.",
    }
    errs = validate_output(out)
    if errs:
        raise AssertionError(f"output schema violation: {errs}")
    return out


def validate_output(o):
    errs = []
    req = {"schema_version": int, "source": str, "logical_unit_id": str,
           "n_kernels": int, "window_ms": (int, float), "top_kernels": list,
           "uncertain_time_share": (int, float), "classification": str,
           "prefill_decode_hint": str, "confidence": str}
    for k, ty in req.items():
        if k not in o:
            errs.append(f"missing {k}")
        elif not isinstance(o[k], ty):
            errs.append(f"{k} type {type(o[k]).__name__}")
    if o.get("classification") not in ("COMPUTE_BOUND", "MEMORY_BOUND",
                                       "MIXED", "UNKNOWN"):
        errs.append("classification enum")
    if o.get("prefill_decode_hint") not in ("prefill", "decode", "n/a"):
        errs.append("prefill_decode_hint enum")
    if o.get("confidence") not in ("LOW", "HIGH"):
        errs.append("confidence enum")
    r = o.get("compute_memory_ratio")
    if r is not None and not (0.0 <= r <= 1.0):
        errs.append("compute_memory_ratio range")
    for tk in o.get("top_kernels", []):
        if tk.get("class") not in ("COMPUTE", "MEMORY", "UNCERTAIN"):
            errs.append(f"top_kernel class: {tk.get('name', '?')[:40]}")
            break
        if "basis" not in tk:
            errs.append("top_kernel basis 누락 (근거 명시 의무)")
            break
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trace Analyzer 4-2 커널 분류 (Exp_13)")
    ap.add_argument("--trace", required=True, help="kernel_tracer 트레이스 파일")
    ap.add_argument("--logical-unit-id", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--workload-type", default=None,
                    choices=["inference", "training", "batch"])
    args = ap.parse_args(argv)
    print(json.dumps(summarize(args.trace, args.logical_unit_id, args.top,
                               workload_type=args.workload_type),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
