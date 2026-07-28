# profiler/collector — CUPTI Activity 커널 트레이서

Trace Analyzer 의 **수집 계층**. `kernel_tracer.so`를 대상 프로세스에 `LD_PRELOAD` 하면
CUPTI Activity API 로 커널/memcpy/memset 레코드를 파이프 구분 텍스트로 흘려보낸다.
`kernel_class/classify.py`(4-2)와 `phase_online/detector.py`(온라인 위상감지)의 입력 공급원.

> **정본 안내**: 이 디렉터리가 정본이다 (Exp_24 승격, 원본은
> `reports/Exp_19_online_phase_230/kernel_tracer.c` — 실험 스냅샷으로 보존, 수정 금지).
> 신규 사용자는 반드시 이쪽(`profiler/collector/`)을 사용할 것.

## 빌드

```bash
make          # gcc -shared -fPIC -O2 kernel_tracer.c -o kernel_tracer.so -lcupti
```

요구사항: CUDA toolkit 의 CUPTI (`/usr/local/cuda/extras/CUPTI`).

## 사용

```bash
KTRACE_OUT=/tmp/trace.txt KTRACE_FLUSH_MS=100 KTRACE_BUF_KB=64 \
  LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64 \
  LD_PRELOAD=$PWD/kernel_tracer.so python3 workload.py
```

`libcupti.so.12`가 기본 검색 경로에 없으면 `LD_LIBRARY_PATH`가 필수다 (230 환경 확인).
`LD_PRELOAD`는 대상 프로세스에만 걸 것 — `timeout` 등 래퍼에 걸리면 로드 실패.

| 환경변수 | 기본 | 설명 |
|---|---|---|
| `KTRACE_OUT` | stderr | 트레이스 출력 파일 |
| `KTRACE_DEFER` | (off) | `1`이면 시작 시 수집 비활성 — `ctypes.CDLL(None).ktrace_start()` 호출 시점부터 수집 |
| `KTRACE_FLUSH_MS` | (off) | N ms 주기 `cuptiActivityFlushAll(0)` — 실행 중 스트리밍 관측 |
| `KTRACE_BUF_KB` | 8192 | activity 버퍼 크기(KB). 스트리밍 시 저율 워크로드는 64 권장 |

## 출력 포맷 (classify.py / phase_online 입력과 동일)

```
K|start_ns|dur_ns|gx,gy,gz|bx,by,bz|stream|regs|smem|name(mangled)
M|start_ns|dur_ns|bytes|copyKind
S|start_ns|dur_ns|bytes
#dropped|N        (드롭 발생 시에만 — 정상이면 이 줄이 없어야 함)
```

## 검증 규격 (Exp_19에서 실측 확정 — 변경 금지)

- **completed-buffer-only 소비**: 주기 플러시는 `flag=0` (완료 버퍼만).
  FORCED(`flag=1`) 주기 플러시는 미완료 레코드가 start=0 으로 유출됨
  (gpt2 실측 39%) — 종료 시(destructor)에만 사용.
- **버퍼 로테이션**: 저율 워크로드 스트리밍은 `KTRACE_BUF_KB=64` — flag=0 은
  완료된 버퍼만 내보내므로 버퍼가 작아야 흐른다.
- **100ms 주기 플러시**: `KTRACE_FLUSH_MS=100` 이 Exp_19 온라인 감지의 검증 조건.
- Exp_19 실측 기준: 2M 레코드 corruption 0 · drop 0, 오버헤드 ≤3.4%.

## 주의

- **LD_PRELOAD 계열 워크로드(libbless 등)와의 조합 금지** — 미해결 행(hang) 이슈
  (Exp_19 report 참조). libbless 대상 트레이싱이 필요하면 별도 검증 선행.
