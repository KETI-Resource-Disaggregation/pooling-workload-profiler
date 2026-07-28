"""알려진 모델 아키텍처 DB (공개 스펙 기반 — 각 항목에 출처 명시).

params 값은 공개 아키텍처 스펙에서 가져온 실제 파라미터 수.
DB 에 없는 모델은 estimator 가 params annotation + 스케일링 휴리스틱으로
아키텍처를 역산한다 (confidence LOW).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelArch:
    name: str
    params: int
    family: str                 # "transformer-decoder" | "transformer-encoder" | "cnn"
    layers: Optional[int] = None
    hidden: Optional[int] = None
    heads: Optional[int] = None
    kv_heads: Optional[int] = None   # GQA. None → heads 와 동일(MHA)
    max_seqlen: Optional[int] = None
    source: str = ""


_M = ModelArch

MODEL_DB = {
    # GPT-2 family — Radford et al. 2019 "Language Models are Unsupervised Multitask Learners"
    "gpt2":        _M("gpt2", 124_000_000, "transformer-decoder", 12, 768, 12,
                      max_seqlen=1024, source="Radford et al. 2019 (GPT-2 small)"),
    "gpt2-medium": _M("gpt2-medium", 355_000_000, "transformer-decoder", 24, 1024, 16,
                      max_seqlen=1024, source="Radford et al. 2019"),
    "gpt2-large":  _M("gpt2-large", 774_000_000, "transformer-decoder", 36, 1280, 20,
                      max_seqlen=1024, source="Radford et al. 2019"),
    # LLaMA — Touvron et al. 2023 "LLaMA: Open and Efficient Foundation Language Models"
    "llama-7b":    _M("llama-7b", 6_740_000_000, "transformer-decoder", 32, 4096, 32,
                      max_seqlen=2048, source="Touvron et al. 2023 (LLaMA-1 6.7B)"),
    "llama-13b":   _M("llama-13b", 13_000_000_000, "transformer-decoder", 40, 5120, 40,
                      max_seqlen=2048, source="Touvron et al. 2023"),
    # Llama-2 — Touvron et al. 2023b (7B/13B 는 MHA, context 4096)
    "llama2-7b":   _M("llama2-7b", 6_740_000_000, "transformer-decoder", 32, 4096, 32,
                      max_seqlen=4096, source="Touvron et al. 2023b (Llama 2)"),
    # OPT — Zhang et al. 2022 "OPT: Open Pre-trained Transformer Language Models"
    # (Exp_12-B 실측 워크로드 — llama-7b 로컬 체크포인트 부재로 동급 아키텍처 대체)
    "opt-6.7b":    _M("opt-6.7b", 6_700_000_000, "transformer-decoder", 32, 4096, 32,
                      max_seqlen=2048, source="Zhang et al. 2022 (OPT 6.7B)"),
    # BERT — Devlin et al. 2018
    "bert-base":   _M("bert-base", 110_000_000, "transformer-encoder", 12, 768, 12,
                      max_seqlen=512, source="Devlin et al. 2018 (BERT-base)"),
    # CNN/ViT — 기존 tests/workload_* 벤치마크와 동일 모델
    "resnet50":    _M("resnet50", 25_600_000, "cnn",
                      source="He et al. 2015; torchvision resnet50 25.6M"),
    "efficientnet-b0": _M("efficientnet-b0", 5_300_000, "cnn",
                      source="Tan&Le 2019; torchvision 5.3M"),
    "vit-base":    _M("vit-base", 86_000_000, "transformer-encoder", 12, 768, 12,
                      max_seqlen=197, source="Dosovitskiy et al. 2020 (ViT-B/16, 224px→197tok)"),
}


def lookup(name: str) -> Optional[ModelArch]:
    if name is None:
        return None
    return MODEL_DB.get(name.strip().lower())


def invert_from_params(params: int) -> ModelArch:
    """params 만 주어진 미등록 decoder LLM 의 아키텍처 역산 (confidence LOW).

    근거 식:
      P ≈ 12·L·H²   (transformer 블록 파라미터 근사 — attention 4H² + FFN 8H²;
                     Kaplan et al. 2020 "Scaling Laws" 의 N≈12·n_layer·d_model²)
      H ≈ 128·L     (GPT-3 175B: H/L=12288/96=128, LLaMA-7B: 4096/32=128 —
                     대표 모델들의 aspect ratio)
    연립하면 L = (P / (12·128²))^(1/3), H = 128·L, head_dim=128 가정(heads=H/128).
    """
    L = max(1, round((params / (12 * 128 * 128)) ** (1.0 / 3.0)))
    H = 128 * L
    heads = max(1, H // 128)
    return ModelArch(
        name=f"unknown-{params // 1_000_000}m",
        params=params,
        family="transformer-decoder",
        layers=L, hidden=H, heads=heads,
        max_seqlen=None,
        source="역산: P≈12LH² (Kaplan et al. 2020) + H/L≈128 가정 — confidence LOW",
    )
