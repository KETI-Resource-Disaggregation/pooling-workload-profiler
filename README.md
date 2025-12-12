# Pooling Workload Profiler

GPU 커널 워크로드 프로파일링 및 분류 시스템

## 개요

딥러닝 워크로드의 GPU 커널을 분석하여 Compute-bound, Memory-bound, Mixed 타입으로 분류합니다.
`pooling-kernel-scheduling`과 연동하여 최적의 스케줄링 결정을 지원합니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              pooling-workload-profiler              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐      ┌──────────────────────┐ │
│  │  kernel_profiler │      │    profiler_api.py   │ │
│  │      (CUDA)     │ ───▶ │     (Port 7070)      │ │
│  └─────────────────┘      └──────────────────────┘ │
│                                    │               │
└────────────────────────────────────┼───────────────┘
                                     │ REST API
                                     ▼
                    ┌────────────────────────────────┐
                    │   pooling-kernel-scheduling    │
                    │    (profiler_client.py)        │
                    └────────────────────────────────┘
```

## 주요 기능

### 1. 커널 타입 분류
- **Compute-bound**: 연산 집약적 (Arithmetic Intensity > 50)
- **Memory-bound**: 메모리 집약적 (Arithmetic Intensity < 10)
- **Mixed**: 혼합 타입 (10 ≤ AI ≤ 50)

### 2. 성능 메트릭 측정
- 실행 시간 (μs)
- 메모리 대역폭 (GB/s)
- Compute Throughput (GFLOPS)
- Arithmetic Intensity (FLOP/byte)

### 3. Co-location 권장
- 커널 타입 기반 최적 조합 권장
- SM 파티션 비율 제안

### 4. REST API 제공
- 프로파일 조회/등록
- Co-location 점수 계산
- `pooling-kernel-scheduling`과 연동

## 시스템 요구사항

- NVIDIA GPU (Compute Capability 7.0+)
- CUDA 12.0+
- Python 3.8+

## 빌드

```bash
# 커널 프로파일러 빌드
make

# 또는 직접
nvcc -O3 -arch=sm_86 -o profiler/kernel_profiler profiler/kernel_profiler.cu -cudart shared
```

## 실행

### 1. API 서버 시작

```bash
# 기본 포트 7070
python3 api/profiler_api.py

# 또는 포트 지정
PROFILER_PORT=7070 python3 api/profiler_api.py
```

### 2. 커널 프로파일러 실행

```bash
cd profiler
./kernel_profiler
```

## 폴더 구조

```
pooling-workload-profiler/
├── profiler/                    # 커널 프로파일러
│   └── kernel_profiler.cu      # CUDA 프로파일러
├── api/                         # REST API
│   └── profiler_api.py         # API 서버
└── docs/                        # 문서
```

## API 엔드포인트

### Health Check
```bash
GET /health
```

### 프로파일 조회
```bash
# 특정 워크로드 프로파일
GET /profile/{workload_id}

# 모든 프로파일
GET /profiles
```

### 프로파일 등록
```bash
# 자동 프로파일링 요청
POST /profile
{
    "workload_id": "my_model",
    "model": "resnet50"
}

# 수동 등록
POST /register
{
    "workload_id": "custom_model",
    "kernel_type": "COMPUTE_BOUND",
    "arithmetic_intensity": 75.0,
    "description": "Custom CNN model"
}
```

### Co-location 권장
```bash
GET /colocation?a=resnet50&b=bert

# Response:
{
    "ok": true,
    "workload_a": {"id": "resnet50", "type": "COMPUTE_BOUND"},
    "workload_b": {"id": "bert", "type": "MIXED"},
    "colocation_score": 0.70,
    "recommendation": "good",
    "sm_partition": {"workload_a": 50, "workload_b": 34}
}
```

## 사전 정의 프로파일

다음 모델들은 사전 정의된 프로파일을 제공합니다:

| 모델 | 커널 타입 | Arithmetic Intensity |
|------|----------|---------------------|
| resnet50 | COMPUTE_BOUND | 85.0 |
| vgg16 | COMPUTE_BOUND | 90.0 |
| bert | MIXED | 35.0 |
| gpt2 | MIXED | 28.0 |
| mobilenet | COMPUTE_BOUND | 65.0 |

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PROFILER_HOST` | 0.0.0.0 | API 호스트 |
| `PROFILER_PORT` | 7070 | API 포트 |

## pooling-kernel-scheduling 연동

`pooling-kernel-scheduling`의 `profiler_client.py`를 통해 연동됩니다:

```python
from profiler_client import ProfilerClient

client = ProfilerClient("http://localhost:7070")

# 프로파일 조회
profile = client.get_profile("resnet50")
print(f"Type: {profile.kernel_type.name}")

# Co-location 점수
rec = client.get_colocation_score("resnet50", "bert")
print(f"Score: {rec.score:.2f}")
print(f"SM Partition: {rec.sm_partition}")
```

## Co-location 점수 기준

| 조합 | 점수 | 권장 |
|------|------|------|
| COMPUTE + MEMORY | 0.95 | 매우 좋음 |
| COMPUTE + MIXED | 0.70 | 좋음 |
| MEMORY + MIXED | 0.60 | 보통 |
| MIXED + MIXED | 0.35 | 주의 |
| COMPUTE + COMPUTE | 0.25 | 비권장 |
| MEMORY + MEMORY | 0.15 | 비권장 |

## 참고 문헌

- Orion: Interference-aware, Interference-free Multi-tenant GPU Scheduling (OSDI'22)
