#!/usr/bin/env python3
"""
Pooling Workload Profiler - REST API Server

워크로드의 커널 타입을 프로파일링하고 결과를 제공하는 API 서버

Endpoints:
- GET  /health              - 서버 상태 확인
- POST /profile             - 워크로드 프로파일링 요청
- GET  /profile/<workload>  - 프로파일링 결과 조회
- GET  /profiles            - 모든 프로파일 조회
- POST /register            - 워크로드 타입 수동 등록

Environment:
- PROFILER_HOST: API 호스트 (default: 0.0.0.0)
- PROFILER_PORT: API 포트 (default: 7070)
"""

import os
import sys
import json
import time
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

# ==================== Configuration ====================
PROFILER_HOST = os.environ.get("PROFILER_HOST", "0.0.0.0")
PROFILER_PORT = int(os.environ.get("PROFILER_PORT", "7070"))

# Profiler binary path
ROOT = Path(__file__).resolve().parents[1]
PROFILER_BIN = ROOT / "profiler" / "kernel_profiler"

# ==================== Kernel Types ====================
class KernelType(Enum):
    UNKNOWN = 0
    COMPUTE_BOUND = 1
    MEMORY_BOUND = 2
    MIXED = 3

# Arithmetic Intensity thresholds
AI_COMPUTE_THRESHOLD = 50.0  # AI > 50 = COMPUTE_BOUND
AI_MEMORY_THRESHOLD = 10.0   # AI < 10 = MEMORY_BOUND

# ==================== Profile Storage ====================
# workload_id -> profile_data
PROFILES: Dict[str, Dict[str, Any]] = {}

# Pre-defined model profiles (based on empirical data)
DEFAULT_PROFILES = {
    "resnet50": {
        "kernel_type": KernelType.COMPUTE_BOUND.name,
        "arithmetic_intensity": 85.0,
        "compute_ratio": 0.70,
        "memory_ratio": 0.15,
        "mixed_ratio": 0.15,
        "description": "CNN model, Conv2D dominant"
    },
    "vgg16": {
        "kernel_type": KernelType.COMPUTE_BOUND.name,
        "arithmetic_intensity": 90.0,
        "compute_ratio": 0.75,
        "memory_ratio": 0.10,
        "mixed_ratio": 0.15,
        "description": "Deep CNN, heavy Conv2D"
    },
    "bert": {
        "kernel_type": KernelType.MIXED.name,
        "arithmetic_intensity": 35.0,
        "compute_ratio": 0.45,
        "memory_ratio": 0.25,
        "mixed_ratio": 0.30,
        "description": "Transformer, Attention + FC"
    },
    "gpt2": {
        "kernel_type": KernelType.MIXED.name,
        "arithmetic_intensity": 28.0,
        "compute_ratio": 0.40,
        "memory_ratio": 0.30,
        "mixed_ratio": 0.30,
        "description": "GPT, Attention + LayerNorm"
    },
    "mobilenet": {
        "kernel_type": KernelType.COMPUTE_BOUND.name,
        "arithmetic_intensity": 65.0,
        "compute_ratio": 0.60,
        "memory_ratio": 0.20,
        "mixed_ratio": 0.20,
        "description": "Lightweight CNN, depthwise conv"
    }
}

# Initialize with default profiles
for name, profile in DEFAULT_PROFILES.items():
    PROFILES[name] = {
        **profile,
        "source": "default",
        "profiled_at": None
    }

# ==================== Profiling Logic ====================
def classify_kernel_type(arithmetic_intensity: float) -> str:
    """Classify kernel type based on Arithmetic Intensity"""
    if arithmetic_intensity > AI_COMPUTE_THRESHOLD:
        return KernelType.COMPUTE_BOUND.name
    elif arithmetic_intensity < AI_MEMORY_THRESHOLD:
        return KernelType.MEMORY_BOUND.name
    else:
        return KernelType.MIXED.name

def compute_colocation_score(type_a: str, type_b: str) -> float:
    """
    Calculate co-location score based on kernel types
    Higher score = better co-location (less interference)

    Based on Orion (OSDI'22) interference model
    """
    scores = {
        (KernelType.COMPUTE_BOUND.name, KernelType.MEMORY_BOUND.name): 0.95,
        (KernelType.MEMORY_BOUND.name, KernelType.COMPUTE_BOUND.name): 0.95,
        (KernelType.COMPUTE_BOUND.name, KernelType.MIXED.name): 0.70,
        (KernelType.MIXED.name, KernelType.COMPUTE_BOUND.name): 0.70,
        (KernelType.MEMORY_BOUND.name, KernelType.MIXED.name): 0.60,
        (KernelType.MIXED.name, KernelType.MEMORY_BOUND.name): 0.60,
        (KernelType.COMPUTE_BOUND.name, KernelType.COMPUTE_BOUND.name): 0.25,
        (KernelType.MEMORY_BOUND.name, KernelType.MEMORY_BOUND.name): 0.15,
        (KernelType.MIXED.name, KernelType.MIXED.name): 0.35,
    }
    return scores.get((type_a, type_b), 0.5)

def recommend_sm_partition(type_a: str, type_b: str, total_sms: int = 84) -> Dict[str, int]:
    """
    Recommend SM partition based on kernel types
    Returns suggested SM allocation for each workload
    """
    # Base allocation (equal split)
    half = total_sms // 2

    if type_a == KernelType.COMPUTE_BOUND.name and type_b == KernelType.MEMORY_BOUND.name:
        # Compute-bound gets more SMs (it benefits more from parallelism)
        return {"workload_a": int(total_sms * 0.6), "workload_b": int(total_sms * 0.4)}
    elif type_a == KernelType.MEMORY_BOUND.name and type_b == KernelType.COMPUTE_BOUND.name:
        return {"workload_a": int(total_sms * 0.4), "workload_b": int(total_sms * 0.6)}
    elif type_a == type_b:
        # Same type: equal split with slight staggering
        return {"workload_a": half, "workload_b": total_sms - half}
    else:
        # Mixed cases: favor the more compute-bound one slightly
        if type_a == KernelType.COMPUTE_BOUND.name:
            return {"workload_a": int(total_sms * 0.55), "workload_b": int(total_sms * 0.45)}
        elif type_b == KernelType.COMPUTE_BOUND.name:
            return {"workload_a": int(total_sms * 0.45), "workload_b": int(total_sms * 0.55)}
        else:
            return {"workload_a": half, "workload_b": total_sms - half}

# ==================== HTTP Handlers ====================
def _json_response(handler: BaseHTTPRequestHandler, obj: Dict, code: int = 200):
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)

def _read_json(handler: BaseHTTPRequestHandler) -> Optional[Dict]:
    try:
        n = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(n) if n > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        return None

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class ProfilerAPI(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/health":
            _json_response(self, {
                "ok": True,
                "service": "pooling-workload-profiler",
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            return

        if path == "/profiles":
            _json_response(self, {
                "ok": True,
                "profiles": PROFILES,
                "count": len(PROFILES)
            })
            return

        if path.startswith("/profile/"):
            workload_id = path.split("/")[-1]
            if workload_id in PROFILES:
                _json_response(self, {
                    "ok": True,
                    "workload": workload_id,
                    "profile": PROFILES[workload_id]
                })
            else:
                _json_response(self, {
                    "ok": False,
                    "error": f"Profile not found: {workload_id}"
                }, 404)
            return

        if path == "/colocation":
            # GET /colocation?a=resnet50&b=bert
            params = {}
            if "?" in self.path:
                query = self.path.split("?")[1]
                for kv in query.split("&"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        params[k] = v

            workload_a = params.get("a", "")
            workload_b = params.get("b", "")

            if not workload_a or not workload_b:
                _json_response(self, {
                    "ok": False,
                    "error": "Both 'a' and 'b' parameters required"
                }, 400)
                return

            profile_a = PROFILES.get(workload_a, {})
            profile_b = PROFILES.get(workload_b, {})

            type_a = profile_a.get("kernel_type", KernelType.UNKNOWN.name)
            type_b = profile_b.get("kernel_type", KernelType.UNKNOWN.name)

            score = compute_colocation_score(type_a, type_b)
            sm_partition = recommend_sm_partition(type_a, type_b)

            _json_response(self, {
                "ok": True,
                "workload_a": {"id": workload_a, "type": type_a},
                "workload_b": {"id": workload_b, "type": type_b},
                "colocation_score": score,
                "recommendation": "good" if score > 0.6 else ("moderate" if score > 0.4 else "poor"),
                "sm_partition": sm_partition
            })
            return

        _json_response(self, {"ok": False, "error": "Not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body = _read_json(self)

        if body is None:
            _json_response(self, {"ok": False, "error": "Invalid JSON"}, 400)
            return

        if path == "/register":
            # Manual registration of workload profile
            workload_id = body.get("workload_id", "").strip()
            kernel_type = body.get("kernel_type", "").upper()
            arithmetic_intensity = float(body.get("arithmetic_intensity", 0))

            if not workload_id:
                _json_response(self, {"ok": False, "error": "workload_id required"}, 400)
                return

            if kernel_type not in [e.name for e in KernelType]:
                # Auto-classify based on AI if type not provided
                kernel_type = classify_kernel_type(arithmetic_intensity)

            PROFILES[workload_id] = {
                "kernel_type": kernel_type,
                "arithmetic_intensity": arithmetic_intensity,
                "compute_ratio": body.get("compute_ratio", 0.33),
                "memory_ratio": body.get("memory_ratio", 0.33),
                "mixed_ratio": body.get("mixed_ratio", 0.34),
                "description": body.get("description", ""),
                "source": "manual",
                "profiled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            _json_response(self, {
                "ok": True,
                "workload": workload_id,
                "profile": PROFILES[workload_id]
            })
            return

        if path == "/profile":
            # Request profiling for a workload
            workload_id = body.get("workload_id", "").strip()
            model_name = body.get("model", "").lower()

            if not workload_id and not model_name:
                _json_response(self, {
                    "ok": False,
                    "error": "workload_id or model required"
                }, 400)
                return

            # Use model name as workload_id if not specified
            if not workload_id:
                workload_id = model_name

            # Check if we have a default profile for this model
            if model_name in DEFAULT_PROFILES:
                profile = {
                    **DEFAULT_PROFILES[model_name],
                    "source": "default",
                    "profiled_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                PROFILES[workload_id] = profile

                _json_response(self, {
                    "ok": True,
                    "workload": workload_id,
                    "profile": profile,
                    "note": "Used pre-defined profile"
                })
                return

            # If no default profile, return unknown type
            profile = {
                "kernel_type": KernelType.UNKNOWN.name,
                "arithmetic_intensity": 0,
                "compute_ratio": 0.33,
                "memory_ratio": 0.33,
                "mixed_ratio": 0.34,
                "description": "Unknown model, requires manual profiling",
                "source": "unknown",
                "profiled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            PROFILES[workload_id] = profile

            _json_response(self, {
                "ok": True,
                "workload": workload_id,
                "profile": profile,
                "note": "Model not recognized, using default unknown profile"
            })
            return

        _json_response(self, {"ok": False, "error": "Not found"}, 404)

# ==================== Main ====================
def main():
    print(f"[profiler-api] Starting Pooling Workload Profiler API")
    print(f"[profiler-api] Host: {PROFILER_HOST}:{PROFILER_PORT}")
    print(f"[profiler-api] Pre-loaded profiles: {list(PROFILES.keys())}")

    httpd = ThreadingHTTPServer((PROFILER_HOST, PROFILER_PORT), ProfilerAPI)
    print(f"[profiler-api] Listening on http://{PROFILER_HOST}:{PROFILER_PORT}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[profiler-api] Shutting down...")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
