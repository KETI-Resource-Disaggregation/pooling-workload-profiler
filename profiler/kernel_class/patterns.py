"""커널명 패턴 DB — Compute/Memory 분류 (근거 명시).

★ 분류 근거는 전부 "커널명 패턴 + roofline 논거"다. 실측 연산강도(FLOP/byte)
  계측이 아니므로 각 패턴에 논거를 명시하고, 매칭 안 되는 커널은 값을
  만들지 않고 UNCERTAIN 으로 둔다 (임의 라벨 금지 — Exp_13 제약).
  패턴 출처: Exp_13 실측 트레이스(노드 230, torch 2.9/CUDA 12.9 커널명)
  + Orion(EuroSys'24)의 compute/memory-bound 커널 구분 관점.

매칭 순서가 의미를 가진다:
  1) MEMORY_OVERRIDE — 이름에 gemm 류가 섞여도 구조상 대역폭 지배인 것
     (예: gemv, flash_fwd_splitkv). COMPUTE 보다 먼저 검사.
  2) COMPUTE — dense matmul/conv 타일 커널 (fused epilogue 로 relu 등
     memory 성 이름이 붙어도 GEMM 이 지배 — 예: s1688gemm_relu).
  3) MEMORY — elementwise/copy/normalization/reduction 류.
  4) 미매칭 → UNCERTAIN.
"""

# (패턴, 논거). 패턴은 demangle 된 커널명에 대한 소문자 substring 매칭.
MEMORY_OVERRIDE = [
    ("gemv", "matrix-vector: FLOP/byte ≈ 2/elem_size ≪ 10 — roofline상 대역폭 지배 (decode 의 지배 커널)"),
    ("splitkv", "decode attention(q_len=1): KV 캐시 순회 읽기가 지배, 연산은 벡터 내적 수준"),
]

COMPUTE = [
    ("gemm", "dense matmul 타일 커널(cutlass/cublas/xmma): FLOP/byte ≈ O(tile) ≫ 50"),
    ("fmha", "fused multi-head attention(cutlass): S×S matmul 체인 지배"),
    ("flash_fwd", "flash attention prefill: 블록 단위 QK^T·PV matmul 지배 (splitkv 는 위에서 선분류)"),
    ("flash_bwd", "flash attention backward: matmul 체인 지배"),
    ("conv", "convolution 타일 커널: implicit GEMM 구조"),
    ("implicit_gemm", "cudnn conv = implicit GEMM"),
    ("winograd", "Winograd conv: 변환 후 batched GEMM"),
    ("fprop", "cudnn conv forward 커널 계열"),
    ("dgrad", "cudnn conv data-grad(역전파) 커널 계열"),
    ("wgrad", "cudnn conv weight-grad(역전파) 커널 계열"),
]

MEMORY = [
    ("elementwise", "원소별 연산: FLOP/byte ≈ 1 미만 — 대역폭 지배"),
    ("vectorized", "at::native vectorized_* (elementwise/gather/layer_norm 계열)"),
    ("reduce", "reduction: 원소당 O(1) 연산, 전량 읽기"),
    ("layer_norm", "normalization: 2-pass 읽기/쓰기 지배"),
    ("layernorm", "동상"),
    ("batch_norm", "동상 (batchnorm)"),
    ("bn_fw", "cudnn batchnorm forward"),
    ("bn_bw", "cudnn batchnorm backward"),
    ("softmax", "row 단위 reduction+scale — 대역폭 지배"),
    ("batchedcopy", "텐서 복사/concat (CatArrayBatchedCopy — KV append 등)"),
    ("copy", "복사 커널"),
    ("nchwtonhwc", "레이아웃 변환 = 순수 데이터 이동"),
    ("nhwctonchw", "동상"),
    ("transpose", "동상"),
    ("im2col", "conv 입력 전개 = 데이터 이동"),
    ("col2im", "동상"),
    ("pooling", "pooling: 원소당 비교/합 — 대역폭 지배"),
    ("pool", "동상"),
    ("index", "index/masked 접근 — gather/scatter 계열"),
    ("gather", "불규칙 읽기 — 대역폭/latency 지배"),
    ("scatter", "불규칙 쓰기 — 동상"),
    ("embedding", "테이블 lookup = gather"),
    ("dropout", "마스크 생성+원소별 곱"),
    ("distribution", "난수 초기화(elementwise)"),
    ("multi_tensor_apply", "fused optimizer(Adam 등): 파라미터 전량 순회 elementwise"),
    ("adam", "optimizer step: 원소별 갱신"),
    ("fill", "상수 채우기 = 쓰기 대역폭"),
    ("arange", "동상"),
    ("triu", "마스크 생성"),
    ("cumsum", "prefix-sum: 스캔(대역폭 지배)"),
    ("argmax", "reduction"),
    ("topk", "부분 정렬 — 읽기 지배"),
    ("sort", "동상"),
    ("compare", "원소별 비교"),
    ("where", "원소별 선택"),
    ("clamp", "원소별 절단"),
    ("cross_entropy", "row reduction + elementwise"),
    ("nll_loss", "동상"),
]


def classify_kernel_name(name: str):
    """커널명 → ("COMPUTE"|"MEMORY"|"UNCERTAIN", 근거 문자열).

    demangle 된 이름을 기대하지만 mangled 이름도 substring 이 대부분 보존돼
    동작한다 (cutlass/cudnn/cublas 심볼은 타입명이 문자열에 포함됨).
    """
    low = name.lower()
    for pat, why in MEMORY_OVERRIDE:
        if pat in low:
            return "MEMORY", f"패턴 '{pat}' (override): {why}"
    for pat, why in COMPUTE:
        if pat in low:
            return "COMPUTE", f"패턴 '{pat}': {why}"
    for pat, why in MEMORY:
        if pat in low:
            return "MEMORY", f"패턴 '{pat}': {why}"
    return "UNCERTAIN", "패턴 미매칭 — 근거 부족, 분류하지 않음"
