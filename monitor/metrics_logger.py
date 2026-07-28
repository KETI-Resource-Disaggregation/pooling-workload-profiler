#!/usr/bin/env python3
# metrics_logger.py
# shm에서 실시간 통계를 읽어 CSV / stdout으로 기록
#
# 측정 항목:
#   - exec_rate_pct : 구간 내 실행 시간 비율 (실제 time-share)
#   - wait_rate_pct : 구간 내 대기 시간 비율
#   - fairness      : 실제 exec 비율 vs 할당 weight 비율 (1.0이 공정)
#   - killer_rate   : 초당 killer op 수
#   - mem_mb        : 현재 GPU 메모리 사용량
#   - gate_state    : 현재 gate 상태

import time
import json
import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "shm"))
from prism_shm import (
    open_shm, MAX_TENANTS,
    MODE_FREE, MODE_OVERCOMMIT, MODE_PROFILING,
    GATE_RUNNING, GATE_KILLER_ACTIVE, GATE_WAITING,
)

_MODE_STR  = {MODE_FREE: "FREE", MODE_OVERCOMMIT: "OVERCOMMIT",
              MODE_PROFILING: "PROFILING"}
_GATE_STR  = {GATE_RUNNING: "RUNNING", GATE_KILLER_ACTIVE: "KILLER_ACTIVE",
              GATE_WAITING: "WAITING"}


class MetricsLogger:
    def __init__(self,
                 group_id:   str,
                 output:     str   = "metrics.csv",
                 interval_s: float = 1.0,
                 stdout:     bool  = False):
        self.shm, self._mm = open_shm(group_id)
        self.output        = output
        self.interval      = interval_s
        self.stdout        = stdout
        self._prev: dict[int, dict] = {}   # tenant_idx → 직전 스냅샷 값

    # ── 스냅샷 ────────────────────────────────────────────────────────────────
    def snapshot(self) -> list[dict]:
        shm  = self.shm
        now  = time.time()
        mode = _MODE_STR.get(shm.mode, f"UNK({shm.mode})")
        rows = []

        # 전체 active 테넌트의 실제 exec 합산 (공정성 분모)
        total_exec_delta = 0
        deltas: dict[int, int] = {}

        for i in range(MAX_TENANTS):
            if shm.alloc[i].tenant_id == b"":
                continue
            t  = shm.tenants[i]
            prev_exec = self._prev.get(i, {}).get("total_exec_us", t.total_exec_us)
            deltas[i] = max(0, t.total_exec_us - prev_exec)
            total_exec_delta += deltas[i]

        for i in range(MAX_TENANTS):
            if shm.alloc[i].tenant_id == b"":
                continue
            t  = shm.tenants[i]
            al = shm.alloc[i]

            prev       = self._prev.get(i, {})
            exec_delta = deltas[i]
            wait_delta = max(0, t.total_wait_us - prev.get("total_wait_us", t.total_wait_us))
            kill_delta = max(0, t.killer_count  - prev.get("killer_count",  t.killer_count))

            window_us = self.interval * 1e6
            exec_rate = exec_delta / window_us * 100  if window_us > 0 else 0.0
            wait_rate = wait_delta / window_us * 100  if window_us > 0 else 0.0

            # 공정성: 실제 exec 비율 / 할당 weight 비율
            weight      = shm.policy.weights[i]
            total_w     = sum(shm.policy.weights[j]
                              for j in range(MAX_TENANTS)
                              if shm.alloc[j].tenant_id != b"") or 1.0
            weight_ratio = weight / total_w
            actual_ratio = (exec_delta / total_exec_delta
                            if total_exec_delta > 0 else 0.0)
            fairness     = (actual_ratio / weight_ratio
                            if weight_ratio > 0 else 0.0)

            row = {
                "ts":           round(now, 3),
                "tenant":       al.tenant_id.decode().rstrip("\x00"),
                "mode":         mode,
                "gate_state":   _GATE_STR.get(t.gate_state, str(t.gate_state)),
                "active":       bool(t.active),
                "exec_rate_pct": round(exec_rate, 2),
                "wait_rate_pct": round(wait_rate, 2),
                "fairness":      round(fairness, 3),
                "killer_rate":   round(kill_delta / self.interval, 2),
                "time_remain_us": t.time_remain_us,
                "total_exec_us": t.total_exec_us,
                "total_wait_us": t.total_wait_us,
                "killer_count":  t.killer_count,
                "mem_mb":        t.mem_used_bytes // (1024 * 1024),
                "virtual_sm":    al.virtual_sm,
                "weight":        round(weight, 4),
                "round_id":      t.round_id,
            }
            rows.append(row)

            # 다음 구간을 위한 이전 값 저장
            self._prev[i] = {
                "total_exec_us": t.total_exec_us,
                "total_wait_us": t.total_wait_us,
                "killer_count":  t.killer_count,
            }

        return rows

    # ── 실행 루프 ─────────────────────────────────────────────────────────────
    def run(self):
        Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output, "w", newline="") as f:
            writer = None
            print(f"[monitor] 시작: group={self.shm.magic:#010x} "
                  f"interval={self.interval}s → {self.output}")

            while True:
                rows = self.snapshot()
                if not rows:
                    time.sleep(self.interval)
                    continue

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()

                for row in rows:
                    writer.writerow(row)

                if self.stdout:
                    self._print_table(rows)

                f.flush()
                time.sleep(self.interval)

    # ── stdout 출력 (개발용) ──────────────────────────────────────────────────
    def _print_table(self, rows: list[dict]):
        ts = rows[0]["ts"]
        mode = rows[0]["mode"]
        print(f"\n[{ts:.1f}] mode={mode}")
        header = f"{'tenant':>8}  {'exec%':>6}  {'wait%':>6}  "
        header += f"{'fair':>6}  {'kill/s':>7}  {'mem_mb':>7}  {'gate':>14}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['tenant']:>8}  "
                  f"{r['exec_rate_pct']:>6.1f}  "
                  f"{r['wait_rate_pct']:>6.1f}  "
                  f"{r['fairness']:>6.3f}  "
                  f"{r['killer_rate']:>7.1f}  "
                  f"{r['mem_mb']:>7}  "
                  f"{r['gate_state']:>14}")

    # ── 단일 JSON 스냅샷 (외부 호출용) ───────────────────────────────────────
    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2)


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="metrics_logger")
    parser.add_argument("--group",    default="default")
    parser.add_argument("--output",   default="metrics.csv")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stdout",   action="store_true",
                        help="CSV 외에 터미널에도 출력")
    parser.add_argument("--once",     action="store_true",
                        help="단일 스냅샷 JSON 출력 후 종료")
    args = parser.parse_args()

    logger = MetricsLogger(args.group, args.output, args.interval, args.stdout)

    if args.once:
        print(logger.snapshot_json())
    else:
        logger.run()
