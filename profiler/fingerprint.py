# fingerprint.py
# 워크로드 지문 계산 + killer_policy 캐시 관리
# 같은 컨테이너가 재진입 시 solo 프로파일링 생략

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

CACHE_DIR     = Path(__file__).parent / "profiles" / "cache"
FINGERPRINT_N = 50   # 지문 계산에 사용할 고유 커널 이름 수


def compute(kernel_names: list[str]) -> str:
    """
    커널 이름 시퀀스에서 워크로드 지문(SHA256 앞 12자)을 계산한다.
    순서를 반영하되 중복 제거 후 첫 N개 사용.
    """
    seen   = []
    unique = []
    for name in kernel_names:
        if name not in seen:
            seen.append(name)
            unique.append(name)
        if len(unique) >= FINGERPRINT_N:
            break

    key = "|".join(unique)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def cache_path(fp: str) -> Path:
    return CACHE_DIR / f"killer_policy_{fp}.json"


def load_cached(fp: str) -> Optional[dict]:
    """캐시에 해당 지문의 policy가 있으면 반환, 없으면 None."""
    p = cache_path(fp)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_cache(policy: dict) -> None:
    """killer_policy를 fingerprint 기반 캐시에 저장."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = policy.get("fingerprint")
    if not fp:
        raise ValueError("policy에 fingerprint 필드가 없습니다")
    with open(cache_path(fp), "w") as f:
        json.dump(policy, f, indent=2)


def list_cache() -> list[dict]:
    """캐시된 모든 policy 목록 반환."""
    if not CACHE_DIR.exists():
        return []
    result = []
    for p in CACHE_DIR.glob("killer_policy_*.json"):
        with open(p) as f:
            result.append(json.load(f))
    return result
