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

### 컨테이너 주입 (Exp_131)

컨테이너에서는 `LD_LIBRARY_PATH` 대신 **`libcupti.so.12` 를 `LD_PRELOAD` 에 함께
싣는다** — 워크로드가 이미 쓰고 있을 `LD_LIBRARY_PATH` 를 덮어쓰지 않기 위함이다
(주입 주체인 device-plugin Allocate 는 파드의 기존 env 를 읽을 수 없다).

```bash
LD_PRELOAD=/…/libcupti.so.12:/…/kernel_tracer.so \
KTRACE_OUT=/tmp/trace.%p.txt KTRACE_FLUSH_MS=100 KTRACE_BUF_KB=64 …
```

의존은 `libcupti.so.12` 하나뿐이다(`ldd` 실측 — 나머지는 glibc). CUPTI 디렉터리
전체를 마운트하지 않아도 되지만, 라이브 경로는 디렉터리째 read-only 로 붙인다.

**`%p` 는 pid 로 치환된다(Exp_131).** `LD_PRELOAD` 로 주입하면 컨테이너의 모든
프로세스가 tracer 를 적재하고, tracer 는 `KTRACE_OUT` 을 `"w"` 로 연다 — 같은
경로를 쓰면 마지막 프로세스(예: 파드 command 의 `cat`)가 앞서 수집한 트레이스를
**길이 0 으로 잘라낸다.** CUDA 를 안 쓰는 프로세스는 빈 파일만 남기므로 수집기는
가장 큰 파일을 고른다. `%p` 가 없으면 종전 동작 그대로다.

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

- ~~LD_PRELOAD 계열 워크로드(libbless 등)와의 조합 금지~~ — **해소됨.**
  Exp_29 에서 이미 해결됐으나 이 문서가 갱신되지 않아 Exp_125·126 이 이 문장을
  근거로 판단을 미뤘다(T-7 사례). Exp_131 에서 232 컨테이너 실측:
  `libbless.so:libcupti.so.12:kernel_tracer.so` 공존 정상, 행 없음, drop 0,
  tracer 추가 비용 **−1.28%**(중앙값, n=5, 1회차 제외 — T-8).
  순서(libbless 앞/뒤)는 회차 산포 안이라 **무관**하나, 기존 항목을 뒤로 밀지
  않는다는 이유로 주입은 `libbless → libcupti → tracer` 순서로 고정한다.
- **`ldd` 는 이 상태에서 행(hang)한다** (Exp_131 1-A). 로더 추적 모드가
  tracer 생성자를 돌리기 때문. 진단 시 `LD_PRELOAD` 를 뺀 채 `ldd` 를 쓴다.
  워크로드 경로(sh/python/CUDA)는 정상이다.
- **기록량이 오버헤드보다 큰 제약이다** (Exp_131 2-D): bert b8 기준
  약 **3.8 MB/s**(15초에 62 MB, 파드당 시간당 ~13.7 GB). 상시 적재 판단은
  성능이 아니라 이 양으로 한다.
