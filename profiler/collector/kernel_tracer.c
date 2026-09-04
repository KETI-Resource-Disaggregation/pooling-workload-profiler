// kernel_tracer.c — Exp_14 확장판 (원본: Exp_13_kernel_class_230/kernel_tracer.c)
//
// Exp_13 CUPTI Activity 트레이서에 KTRACE_FLUSH_MS 주기 플러시 스레드 추가.
// 4-3/4-4 는 "지속 시계열"이 입력 — 사후 파일이 아니라 실행 중 주기적으로
// 레코드가 파일로 흘러나오는지(스트리밍 가능성)를 Phase 0 에서 점검한다.
//
// 출력(KTRACE_OUT, 기본 stderr): 파이프 구분 텍스트, 한 줄 = 레코드 1개
//   K|start_ns|dur_ns|gridX,gridY,gridZ|blockX,blockY,blockZ|stream|regs|smem|name(mangled)
//   M|start_ns|dur_ns|bytes|copyKind         (memcpy)
//   S|start_ns|dur_ns|bytes                  (memset)
// 환경변수:
//   KTRACE_OUT=/path    출력 파일
//   KTRACE_DEFER=1      시작 시 수집 비활성 — 프로세스 내에서
//                       ctypes 등으로 ktrace_start() 호출 시점부터 수집
//                       (prefill 제외한 decode 정상상태만 관측하는 용도)
//   KTRACE_FLUSH_MS=N   N ms 마다 cuptiActivityFlushAll(0)+fflush — 실행 중
//                       스트리밍 관측 (미설정 시 Exp_13 과 동일: 종료 시 플러시)
//   KTRACE_BUF_KB=N     activity 버퍼 크기(KB, 기본 8192). 스트리밍 시 저율
//                       워크로드는 작게(예: 64) — flag=0 은 완료 버퍼만 내보냄
//
// 빌드: gcc -shared -fPIC -O2 kernel_tracer.c -o kernel_tracer.so \
//         -I/usr/local/cuda/extras/CUPTI/include \
//         -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <cupti.h>

/* 기본 8MB. 스트리밍 용도는 KTRACE_BUF_KB 로 줄인다 — flag=0 플러시는
 * "완료된 버퍼"만 내보내므로 저율 워크로드는 버퍼가 작아야 흐른다.
 * (FORCED 플러시는 미완료 레코드를 start=0 으로 내보냄 — Exp_14 실측 39% → 미사용) */
static size_t buf_size = 8 * 1024 * 1024;

static FILE *out = NULL;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int enabled = 0;

#define CUPTI_CHECK(call) do { \
    CUptiResult _s = (call); \
    if (_s != CUPTI_SUCCESS) { \
        const char *e; cuptiGetResultString(_s, &e); \
        fprintf(stderr, "[ktrace] %s failed: %s\n", #call, e); \
    } \
} while (0)

static void CUPTIAPI buf_requested(uint8_t **buffer, size_t *size,
                                   size_t *max_records) {
    *size = buf_size;
    *buffer = (uint8_t *)malloc(buf_size);
    *max_records = 0;
}

static void CUPTIAPI buf_completed(CUcontext ctx, uint32_t stream_id,
                                   uint8_t *buffer, size_t size,
                                   size_t valid_size) {
    CUpti_Activity *rec = NULL;
    pthread_mutex_lock(&mu);
    for (;;) {
        CUptiResult st = cuptiActivityGetNextRecord(buffer, valid_size, &rec);
        if (st != CUPTI_SUCCESS) break;   /* MAX_LIMIT_REACHED 포함 */
        switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)rec;
            fprintf(out, "K|%llu|%llu|%d,%d,%d|%d,%d,%d|%u|%d|%d|%s\n",
                    (unsigned long long)k->start,
                    (unsigned long long)(k->end - k->start),
                    k->gridX, k->gridY, k->gridZ,
                    k->blockX, k->blockY, k->blockZ,
                    k->streamId, (int)k->registersPerThread,
                    (int)(k->staticSharedMemory + k->dynamicSharedMemory),
                    k->name ? k->name : "?");
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)rec;
            fprintf(out, "M|%llu|%llu|%llu|%d\n",
                    (unsigned long long)m->start,
                    (unsigned long long)(m->end - m->start),
                    (unsigned long long)m->bytes, (int)m->copyKind);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            CUpti_ActivityMemset4 *m = (CUpti_ActivityMemset4 *)rec;
            fprintf(out, "S|%llu|%llu|%llu\n",
                    (unsigned long long)m->start,
                    (unsigned long long)(m->end - m->start),
                    (unsigned long long)m->bytes);
            break;
        }
        default:
            break;
        }
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped);
    if (dropped)
        fprintf(out, "#dropped|%llu\n", (unsigned long long)dropped);
    pthread_mutex_unlock(&mu);
    free(buffer);
}

static void enable_kinds(void) {
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    enabled = 1;
}

/* KTRACE_DEFER=1 일 때 워크로드가 관측 시작 시점을 지정하는 진입점
 * (python: ctypes.CDLL(None).ktrace_start()) */
void ktrace_start(void) {
    if (!enabled) {
        enable_kinds();
        fprintf(out, "#start_deferred\n");
        fflush(out);
    }
}

void ktrace_stop(void) {
    if (enabled) {
        cuptiActivityFlushAll(1);
        CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
        enabled = 0;
    }
}

/* KTRACE_FLUSH_MS 주기 플러시 스레드 — 실행 중 스트리밍 관측용 */
static volatile int flusher_run = 0;
static pthread_t flusher_tid;

static void *flusher(void *arg) {
    long ms = (long)(intptr_t)arg;
    while (flusher_run) {
        struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
        nanosleep(&ts, NULL);
        if (!flusher_run) break;
        /* flag=0: 완료된 버퍼만 — 미완료 레코드(start=0) 유출 없음.
         * 저율 워크로드는 KTRACE_BUF_KB 로 버퍼를 줄여 흐르게 한다.
         * (FORCED 는 gpt2 실측에서 레코드 39% 가 start=0 — 채택 안 함) */
        cuptiActivityFlushAll(0);
        pthread_mutex_lock(&mu);
        if (out) fflush(out);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

// [Exp_131] KTRACE_OUT 안의 "%p" 를 pid 로 치환한다.
//   ★근거: LD_PRELOAD 로 주입하면 컨테이너의 **모든** 프로세스가 이 생성자를
//   돈다. 파드 command 가 `bash -c "python ...; cat ..."` 이면 bash·python·cat
//   셋이 같은 KTRACE_OUT 을 fopen(...,"w") 로 열어, 마지막 프로세스가 앞서
//   수집한 트레이스를 **길이 0 으로 잘라낸다**(Exp_131 1-C 에서 확인).
//   CUDA 를 쓰지 않는 프로세스는 빈 파일만 남기므로 수집기는 가장 큰 것을 고른다.
//   "%p" 가 없으면 종전과 동일하게 동작한다(Exp_19 스크립트 무영향).
static void ktrace_expand_pid(const char *in, char *dst, size_t n) {
    const char *m = strstr(in, "%p");
    if (!m) { snprintf(dst, n, "%s", in); return; }
    snprintf(dst, n, "%.*s%ld%s", (int)(m - in), in, (long)getpid(), m + 2);
}

__attribute__((constructor)) static void ktrace_init(void) {
    const char *path = getenv("KTRACE_OUT");
    char resolved[4096];
    if (path) {
        ktrace_expand_pid(path, resolved, sizeof resolved);
        out = fopen(resolved, "w");
    } else {
        out = stderr;
    }
    if (!out) out = stderr;
    const char *bk = getenv("KTRACE_BUF_KB");
    if (bk && atol(bk) >= 16)
        buf_size = (size_t)atol(bk) * 1024;
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(buf_requested, buf_completed));
    const char *defer = getenv("KTRACE_DEFER");
    if (!(defer && defer[0] == '1'))
        enable_kinds();
    const char *fl = getenv("KTRACE_FLUSH_MS");
    if (fl && atol(fl) > 0) {
        flusher_run = 1;
        pthread_create(&flusher_tid, NULL, flusher, (void *)(intptr_t)atol(fl));
    }
}

__attribute__((destructor)) static void ktrace_fini(void) {
    if (flusher_run) {
        flusher_run = 0;
        pthread_join(flusher_tid, NULL);
    }
    cuptiActivityFlushAll(1);
    if (out && out != stderr) fclose(out);
}
