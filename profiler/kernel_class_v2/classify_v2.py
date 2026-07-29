#!/usr/bin/env python3
"""분류엔진 v2 — 공간(SM occupancy) 차원 추가 (Exp_48).

v1(kernel_class/, 커널명 시간 점유 C/M) 정본 무변경 — 본 모듈은 병행 v2.
같은 트레이스(K|start|dur|gx,gy,gz|bx,by,bz|stream|regs|smem|name — 수집기가
이미 방출, Exp_48 Phase 0 확인)를 자체 파싱해 공간 지표를 추가한다.

v2 지표 (전부 커널 시간 가중 평균, 물리 근거 명시):
  cm_ratio        v1 그대로 (patterns.classify_kernel_name 재사용 — 시간 축)
  sm_occupancy    이론 occupancy = active_warps/max_warps_per_SM.
                  ★공식 구현: cuda_occupancy.h cudaOccMaxActiveBlocksPerMultiprocessor
                  (occ_calc 바이너리 — 임의 근사 없음, launch(block,regs,smem) 실값)
  grid_saturation min(1, grid_blocks / SM수) — 그리드가 SM 을 덮는가
  device_fill     min(blocks, maxActiveBlk×SM수)×warps_per_block / (SM수×max_warps)
                  — 장치 전체 워프 용량 중 실제 점유 비율 (occupancy×coverage 통합)
클래스별 분해: v1 클래스(COMPUTE/MEMORY/UNCERTAIN)별 device_fill —
  "MEMORY 커널이 SM 을 얼마나 점유하는가"(Exp_17 페어링 질문)에 직접 답한다.

usage: classify_v2.py --trace <path> [--device 1] [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kernel_class.patterns import classify_kernel_name  # v1 재사용 (읽기 전용)

HERE = os.path.dirname(os.path.abspath(__file__))
OCC_CALC = os.path.join(HERE, "occ_calc")


def parse_trace_v2(path):
    """K 레코드 → dur/blocks/threads/regs/smem/name (v1 파서는 block 차원을 버림)."""
    import gzip
    kernels = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith("K|"):
                continue
            p = line.rstrip("\n").split("|")
            if len(p) < 9:
                continue
            gx, gy, gz = (int(x) for x in p[3].split(","))
            bx, by, bz = (int(x) for x in p[4].split(","))
            kernels.append({
                "start_ns": int(p[1]),
                "dur_ns": int(p[2]), "blocks": gx * gy * gz,
                "threads": bx * by * bz, "regs": int(p[6]),
                "smem": int(p[7]), "name": p[8],
            })
    return kernels


def occ_lookup(launches, device):
    """유닉 launch (threads,regs,smem) → (max_blocks_per_sm, occupancy). 공식 계산기 일괄 호출."""
    uniq = sorted(set(launches))
    inp = "\n".join(f"{t} {r} {s}" for t, r, s in uniq) + "\n"
    out = subprocess.run([OCC_CALC, str(device)], input=inp,
                        capture_output=True, text=True, check=True).stdout.splitlines()
    hdr = out[0].split("|")  # DEV|name|sm|max_warps|warp
    dev = {"name": hdr[1], "num_sm": int(hdr[2]), "max_warps": int(hdr[3]),
           "warp": int(hdr[4])}
    table = {}
    for key, line in zip(uniq, out[1:]):
        b, w, o = line.split()
        table[key] = (int(b), float(o))
    return dev, table


def classify_v2(trace_path, device=0):
    ks = parse_trace_v2(trace_path)
    if not ks:
        return {"error": "empty trace"}
    dev, occ = occ_lookup([(k["threads"], k["regs"], k["smem"]) for k in ks], device)
    nsm, mw, warp = dev["num_sm"], dev["max_warps"], dev["warp"]

    t0 = min(k["start_ns"] for k in ks)
    t1 = max(k["start_ns"] + k["dur_ns"] for k in ks)
    tot = 0
    agg = defaultdict(float)          # 전체 가중 합
    cls_t = defaultdict(float)        # v1 클래스별 시간
    cls_fill = defaultdict(float)     # v1 클래스별 device_fill 가중 합
    for k in ks:
        d = k["dur_ns"]
        maxblk, o = occ[(k["threads"], k["regs"], k["smem"])]
        wpb = (k["threads"] + warp - 1) // warp
        sat = min(1.0, k["blocks"] / nsm)
        active_blocks = min(k["blocks"], maxblk * nsm) if maxblk else 0
        fill = active_blocks * wpb / (nsm * mw)
        cl, _ = classify_kernel_name(k["name"])
        tot += d
        agg["occ"] += d * o
        agg["sat"] += d * sat
        agg["fill"] += d * fill
        cls_t[cl] += d
        cls_fill[cl] += d * fill

    ct, mt = cls_t.get("COMPUTE", 0), cls_t.get("MEMORY", 0)
    cm = ct / (ct + mt) if ct + mt else None
    out = {
        "schema_version": 2, "source": "kernel_class_v2",
        "device": dev, "n_kernels": len(ks),
        "kernel_time_ms": round(tot / 1e6, 1),
        "cm_ratio": round(cm, 4) if cm is not None else None,   # v1 축 유지
        "sm_occupancy": round(agg["occ"] / tot, 4),
        "grid_saturation": round(agg["sat"] / tot, 4),
        "device_fill": round(agg["fill"] / tot, 4),
        # 벽시계 커널 실행 밀도 (Exp_49 — launch-bound 워크로드 판별 차원):
        # 커널 총 시간 / 트레이스 벽시계. 다중 스트림 중첩 시 >1 가능(중첩 미보정).
        "kernel_density": round(min(tot / (t1 - t0), 1.0), 4) if t1 > t0 else None,
        "wall_fill": round((agg["fill"] / tot) * min(tot / (t1 - t0), 1.0), 4)
                     if t1 > t0 else None,
        "class_share": {c: round(t / tot, 4) for c, t in sorted(cls_t.items())},
        "class_device_fill": {c: round(cls_fill[c] / t, 4)
                              for c, t in sorted(cls_t.items()) if t > 0},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = classify_v2(a.trace, a.device)
    s = json.dumps(r, ensure_ascii=False, indent=2)
    if a.json:
        open(a.json, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
