# online_profiler.py
# 3단계 온라인 프로파일링 파이프라인
#
# 단계 1: 캐시 확인  — 1 iteration 관찰 → fingerprint → 캐시 HIT 시 solo 생략
# 단계 2: 패턴 임시 정책 — 관찰된 커널 이름에 패턴 매칭 즉시 적용
# 단계 3: Solo 프로파일 — PROFILING 모드 진입 → N iter → killer_policy 추출

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "shm"))
from kraken_shm import open_shm, MODE_FREE, MODE_OVERCOMMIT, MODE_PROFILING, \
                     GATE_WAITING, GATE_RUNNING, MAX_TENANTS

from killer_extractor import KernelRecord, KillerPolicy, extract, from_torch_profiler
from fingerprint      import compute as compute_fp, load_cached, save_cache
from pattern_matcher  import match as pattern_match


PROFILING_ITERS_DEFAULT = 3      # solo 프로파일링 iteration 수
PROFILING_TIMEOUT_S     = 30.0   # 기존 테넌트 대기 최대 시간


class OnlineProfiler:
    """
    새 테넌트에 대한 온라인 프로파일링 파이프라인.

    사용 흐름:
        profiler = OnlineProfiler(group_id="default")
        policy   = profiler.run(
            tenant_idx  = 2,
            workload_fn = lambda: model(inputs),
            n_iters     = 3,
        )
    """

    def __init__(self, group_id: str):
        self.shm, self._mm = open_shm(group_id)
        self.group_id      = group_id

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        tenant_idx:  int,
        workload_fn: Callable,
        n_iters:     int   = PROFILING_ITERS_DEFAULT,
        top_pct:     float = 0.20,
    ) -> KillerPolicy:
        """온라인 프로파일링 전체 파이프라인 실행."""

        # 단계 1: 1-iteration 관찰 → fingerprint → 캐시 확인
        observed_names, cached = self._check_cache(workload_fn)
        if cached:
            print(f"[profiler] cache HIT (fp={cached['fingerprint']}) "
                  f"→ solo 프로파일 생략")
            return self._policy_from_dict(cached)

        # 단계 2: 관찰된 커널 이름으로 패턴 매칭 임시 정책 적용
        self._apply_pattern_policy(tenant_idx, observed_names)

        # 단계 3: PROFILING 모드 → solo N-iteration → killer_policy 추출
        policy = self._solo_profile(tenant_idx, workload_fn, n_iters, top_pct)

        return policy

    # ── 단계 1: 캐시 확인 ──────────────────────────────────────────────────────

    def _check_cache(
        self,
        workload_fn: Callable,
    ) -> tuple[list[str], Optional[dict]]:
        """
        1 iteration 비gating 관찰로 fingerprint를 계산하고 캐시를 조회한다.

        반환: (observed_kernel_names, cached_policy_or_None)
        torch가 없는 환경에서는 빈 이름 목록과 None을 반환해 항상 solo profile로 진행.
        """
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
        except ImportError:
            return [], None

        # 1 iteration 관찰 (gating 없음 — shm mode에 관계없이 torch.profiler만 사용)
        with profile(
            activities    = [ProfilerActivity.CUDA],
            record_shapes = False,
            with_stack    = False,
        ) as prof:
            workload_fn()
            torch.cuda.synchronize()

        # CUDA 커널 이름 수집
        names = []
        for event in prof.events():
            if (hasattr(event, "device_type")
                    and str(event.device_type) == "DeviceType.CUDA"
                    and event.cuda_time > 0):
                names.append(event.name)

        if not names:
            return [], None

        fp     = compute_fp(names)
        cached = load_cached(fp)

        if cached:
            return names, cached

        return names, None

    # ── 단계 2: 패턴 매칭 임시 정책 ────────────────────────────────────────────

    def _apply_pattern_policy(
        self,
        tenant_idx:    int,
        kernel_names:  list[str],
    ) -> None:
        """
        관찰된 커널 이름에 패턴 매칭을 수행하고 결과를 hot-reload 경로에 저장.
        shm.killer_policy_version을 bump → runtime이 다음 라운드 시작 시 재로드.
        """
        if not kernel_names:
            print(f"[profiler] 패턴 매칭: 커널 이름 없음 → 기본 패턴 유지")
            return

        result = pattern_match(kernel_names)
        print(f"[profiler] 패턴 매칭 임시 정책 (tenant={tenant_idx}): "
              f"{len(result.killer_names)} killers, "
              f"type={result.workload_type}, confidence=LOW")

        # well-known hot-reload 경로에 저장 (runtime의 HOT_POLICY_PATH_FMT와 일치)
        import json
        policy_dict = {
            "killer_names":     result.killer_names,
            "killer_indices":   result.killer_indices,
            "kernels_per_iter": 0,   # 아직 미확정
            "confidence":       "LOW",
        }
        hot_path = Path(f"/tmp/kraken_killers_{self.group_id}_{tenant_idx}.json")
        with open(hot_path, "w") as f:
            json.dump(policy_dict, f)

        # shm version bump → runtime이 다음 라운드에 reload_killer_policy_if_needed() 수행
        self.shm.killer_policy_version += 1
        print(f"[profiler] hot-reload 경로: {hot_path}  "
              f"(version={self.shm.killer_policy_version})")

    # ── 단계 3: Solo 프로파일 ──────────────────────────────────────────────────

    def _solo_profile(
        self,
        tenant_idx:  int,
        workload_fn: Callable,
        n_iters:     int,
        top_pct:     float,
    ) -> KillerPolicy:
        """PROFILING 모드 진입 → N-iter solo 실행 → policy 추출 → 모드 복귀."""
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
            has_torch = True
        except ImportError:
            has_torch = False

        print(f"[profiler] PROFILING 모드 진입 "
              f"(tenant={tenant_idx}, iters={n_iters})")
        self._enter_profiling_mode(tenant_idx, n_iters)

        try:
            if has_torch:
                policy = self._profile_with_torch(workload_fn, n_iters, top_pct)
            else:
                policy = self._profile_without_torch(workload_fn, n_iters, top_pct)
        finally:
            self._exit_profiling_mode()

        print(f"[profiler] 완료 → killer {len(policy.killer_names)}개 "
              f"({policy.killer_ratio:.1%}), type={policy.workload_type}, "
              f"avg={policy.avg_killer_us:.0f}μs")

        # 정밀 policy를 hot-reload 경로에 저장 + version bump
        self._save_hot_policy(tenant_idx, policy)
        return policy

    def _profile_with_torch(
        self,
        workload_fn: Callable,
        n_iters:     int,
        top_pct:     float,
    ) -> KillerPolicy:
        import torch
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities    = [ProfilerActivity.CUDA],
            record_shapes = False,
            with_stack    = False,
        ) as prof:
            for _ in range(n_iters):
                workload_fn()
                torch.cuda.synchronize()

        return from_torch_profiler(prof, top_pct=top_pct)

    def _profile_without_torch(
        self,
        workload_fn: Callable,
        n_iters:     int,
        top_pct:     float,
    ) -> KillerPolicy:
        """
        torch 없는 환경 폴백:
        runtime이 shm에 기록한 커널 통계를 읽어 policy 추출.
        현재 구현: runtime이 shm에 커널 이름을 기록하는 기능이 없으므로
        패턴 매칭 기반 LOW-confidence policy를 반환.
        """
        for _ in range(n_iters):
            workload_fn()

        # shm에서 killer_count와 exec_us로 근사 policy 생성
        shm = self.shm
        t   = shm.tenants[0]   # solo 모드이므로 첫 슬롯 사용

        from killer_extractor import KillerPolicy
        from fingerprint      import compute as compute_fp
        import time

        fp = compute_fp([f"unknown_{i}" for i in range(10)])
        policy = KillerPolicy(
            schema_version   = 1,
            fingerprint      = fp,
            workload_type    = "UNKNOWN",
            kernels_per_iter = 0,
            killer_indices   = [],
            killer_names     = [],
            killer_ratio     = 0.0,
            top_pct          = top_pct,
            avg_killer_us    = 0.0,
            confidence       = "LOW",
        )
        print("[profiler] torch 없음 → LOW-confidence policy 생성")
        return policy

    # ── hot-reload 저장 ────────────────────────────────────────────────────────

    def _save_hot_policy(self, tenant_idx: int, policy: "KillerPolicy") -> None:
        """KillerPolicy를 hot-reload 경로에 저장하고 shm version을 bump."""
        import json
        policy_dict = {
            "killer_names":     policy.killer_names,
            "killer_indices":   policy.killer_indices,
            "kernels_per_iter": policy.kernels_per_iter,
            "confidence":       policy.confidence,
        }
        hot_path = Path(f"/tmp/kraken_killers_{self.group_id}_{tenant_idx}.json")
        with open(hot_path, "w") as f:
            json.dump(policy_dict, f)
        self.shm.killer_policy_version += 1
        print(f"[profiler] hot-reload 저장: {hot_path}  "
              f"(version={self.shm.killer_policy_version})")

    # ── PROFILING 모드 진입/탈출 ───────────────────────────────────────────────

    def _enter_profiling_mode(self, tenant_idx: int, n_iters: int) -> None:
        """shm PROFILING 모드 설정 후 기존 테넌트들이 killer 경계에 도달할 때까지 대기."""
        shm = self.shm
        shm.profiling_tenant_idx  = tenant_idx
        shm.profiling_iter_remain = n_iters
        shm.mode                  = MODE_PROFILING

        deadline = time.monotonic() + PROFILING_TIMEOUT_S
        while time.monotonic() < deadline:
            all_paused = all(
                not shm.tenants[i].active
                or i == tenant_idx
                or self._is_at_boundary(i)
                for i in range(shm.tenant_count)
            )
            if all_paused:
                break
            time.sleep(0.005)

    def _exit_profiling_mode(self) -> None:
        """PROFILING 모드 해제 → FREE 또는 OVERCOMMIT으로 복귀."""
        shm = self.shm
        shm.profiling_tenant_idx  = -1
        shm.profiling_iter_remain = 0

        virtual_total = sum(
            shm.alloc[i].virtual_sm
            for i in range(shm.tenant_count)
            if shm.tenants[i].active
        )
        new_mode = (MODE_OVERCOMMIT
                    if virtual_total > shm.physical_sm_total
                    else MODE_FREE)
        shm.mode = new_mode
        print(f"[profiler] 모드 복귀 → "
              f"{'OVERCOMMIT' if new_mode == MODE_OVERCOMMIT else 'FREE'}")

    def _is_at_boundary(self, tenant_idx: int) -> bool:
        """killer 실행 중이 아닌 상태 (WAITING 또는 RUNNING = 경계에 있음)."""
        state = self.shm.tenants[tenant_idx].gate_state
        return state in (GATE_WAITING, GATE_RUNNING)

    # ── 유틸 ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _policy_from_dict(d: dict) -> KillerPolicy:
        from killer_extractor import KillerPolicy
        return KillerPolicy(
            schema_version   = d["schema_version"],
            fingerprint      = d["fingerprint"],
            workload_type    = d["workload_type"],
            kernels_per_iter = d["kernels_per_iter"],
            killer_indices   = d["killer_indices"],
            killer_names     = d["killer_names"],
            killer_ratio     = d["killer_ratio"],
            top_pct          = d["top_pct"],
            avg_killer_us    = d["avg_killer_us"],
            confidence       = d.get("confidence", "HIGH"),
            profiled_at      = d.get("profiled_at", ""),
        )
