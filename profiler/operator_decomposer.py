#!/usr/bin/env python3
"""
Operator Decomposer - 워크로드를 오퍼레이터 단위로 분해하고 정량화

워크로드(모델)를 입력받아:
1. torch.profiler로 오퍼레이터 구성 분석
2. 각 오퍼레이터의 실행 시간 비율 계산
3. 6D 메트릭 기반 워크로드 특성 정량화
4. pooling-kernel-scheduling에 전달할 데이터 생성

Usage:
    decomposer = OperatorDecomposer()
    result = decomposer.decompose_model("resnet50", batch_size=16)
    quantified = decomposer.quantify_workload(result)
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading

# ==================== Configuration ====================
PROFILER_ROOT = Path(__file__).resolve().parent
DB_PATH = PROFILER_ROOT / "operator_database.json"

# ==================== Enums ====================
class KernelType(Enum):
    UNKNOWN = "UNKNOWN"
    COMPUTE_BOUND = "COMPUTE_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    MIXED = "MIXED"


class OperatorCategory(Enum):
    """오퍼레이터 카테고리"""
    CONV = "Conv"           # Conv2d, Conv1d, etc.
    LINEAR = "Linear"       # Linear, MatMul
    NORM = "Norm"           # BatchNorm, LayerNorm, etc.
    ACTIVATION = "Activation"  # ReLU, GELU, SiLU, etc.
    POOL = "Pool"           # MaxPool, AvgPool, AdaptivePool
    ATTENTION = "Attention" # MultiheadAttention, ScaledDotProduct
    ELEMENTWISE = "Elementwise"  # Add, Mul, Cat, etc.
    EMBEDDING = "Embedding" # Embedding layers
    OTHER = "Other"


# ==================== Data Classes ====================
@dataclass
class OperatorProfile:
    """오퍼레이터 6D 프로파일"""
    name: str
    category: str
    sm_util: float = 0.0          # SM 활용률 (%)
    l2_throughput: float = 0.0    # L2 캐시 처리량 (%)
    l1_throughput: float = 0.0    # L1 캐시 처리량 (%)
    dram_util: float = 0.0        # DRAM 대역폭 활용 (%)
    ipc: float = 0.0              # Instructions Per Cycle
    pipeline_util: float = 0.0    # FMA 파이프라인 활용 (%)
    kernel_type: str = "UNKNOWN"
    confidence: str = "medium"    # high, medium, low
    source: str = "database"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OperatorComposition:
    """오퍼레이터 구성 정보"""
    name: str
    category: str
    time_us: float = 0.0
    time_percent: float = 0.0
    call_count: int = 0
    avg_time_us: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkloadProfile:
    """워크로드 전체 프로파일"""
    workload_id: str
    model_name: str
    batch_size: int
    mode: str  # "inference" or "training"

    # 오퍼레이터 구성
    total_time_ms: float = 0.0
    operators: Dict[str, OperatorComposition] = field(default_factory=dict)

    # 집계된 6D 메트릭
    sm_util: float = 0.0
    l2_throughput: float = 0.0
    l1_throughput: float = 0.0
    dram_util: float = 0.0
    ipc: float = 0.0
    pipeline_util: float = 0.0

    # 분류
    kernel_type: str = "UNKNOWN"
    arithmetic_intensity: float = 0.0

    # 메타데이터
    profiled_at: str = ""
    coverage: float = 0.0  # DB 커버리지 (%)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['operators'] = {k: v.to_dict() if hasattr(v, 'to_dict') else v
                         for k, v in self.operators.items()}
        return d


# ==================== Operator Database ====================
class OperatorDatabase:
    """오퍼레이터 6D 프로파일 데이터베이스"""

    # 기본 오퍼레이터 프로파일 (NCU 측정 + 분석 기반)
    DEFAULT_PROFILES = {
        # CNN 오퍼레이터 (NCU 측정 기반)
        "Conv2d": OperatorProfile(
            name="Conv2d", category="Conv",
            sm_util=54.5, l2_throughput=44.9, l1_throughput=35.5,
            dram_util=64.0, ipc=0.32, pipeline_util=11.6,
            kernel_type="COMPUTE_BOUND", confidence="high", source="NCU"
        ),
        "BatchNorm2d": OperatorProfile(
            name="BatchNorm2d", category="Norm",
            sm_util=17.8, l2_throughput=31.2, l1_throughput=22.0,
            dram_util=84.6, ipc=0.19, pipeline_util=8.0,
            kernel_type="MEMORY_BOUND", confidence="high", source="NCU"
        ),
        "Linear": OperatorProfile(
            name="Linear", category="Linear",
            sm_util=29.0, l2_throughput=25.2, l1_throughput=29.8,
            dram_util=44.0, ipc=0.31, pipeline_util=19.8,
            kernel_type="MIXED", confidence="high", source="NCU"
        ),
        "ReLU": OperatorProfile(
            name="ReLU", category="Activation",
            sm_util=10.0, l2_throughput=41.3, l1_throughput=22.5,
            dram_util=89.8, ipc=0.11, pipeline_util=5.6,
            kernel_type="MEMORY_BOUND", confidence="high", source="NCU"
        ),
        "MaxPool2d": OperatorProfile(
            name="MaxPool2d", category="Pool",
            sm_util=54.2, l2_throughput=34.4, l1_throughput=21.7,
            dram_util=92.4, ipc=0.55, pipeline_util=24.6,
            kernel_type="MEMORY_BOUND", confidence="high", source="NCU"
        ),

        # Elementwise 오퍼레이터 (대역폭 분석 기반)
        "Elementwise": OperatorProfile(
            name="Elementwise", category="Elementwise",
            sm_util=0.3, l2_throughput=53.2, l1_throughput=35.4,
            dram_util=88.6, ipc=0.1, pipeline_util=0.1,
            kernel_type="MEMORY_BOUND", confidence="high", source="bandwidth_analysis"
        ),
        "Add": OperatorProfile(
            name="Add", category="Elementwise",
            sm_util=0.3, l2_throughput=53.2, l1_throughput=35.4,
            dram_util=88.6, ipc=0.1, pipeline_util=0.1,
            kernel_type="MEMORY_BOUND", confidence="high", source="bandwidth_analysis"
        ),

        # Transformer 오퍼레이터 (역산 기반)
        "LayerNorm": OperatorProfile(
            name="LayerNorm", category="Norm",
            sm_util=15.0, l2_throughput=28.0, l1_throughput=18.0,
            dram_util=75.0, ipc=0.15, pipeline_util=6.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),
        "GELU": OperatorProfile(
            name="GELU", category="Activation",
            sm_util=12.0, l2_throughput=35.0, l1_throughput=20.0,
            dram_util=80.0, ipc=0.12, pipeline_util=5.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),
        "Attention": OperatorProfile(
            name="Attention", category="Attention",
            sm_util=45.0, l2_throughput=40.0, l1_throughput=30.0,
            dram_util=55.0, ipc=0.35, pipeline_util=15.0,
            kernel_type="MIXED", confidence="medium", source="estimated"
        ),
        "MultiheadAttention": OperatorProfile(
            name="MultiheadAttention", category="Attention",
            sm_util=45.0, l2_throughput=40.0, l1_throughput=30.0,
            dram_util=55.0, ipc=0.35, pipeline_util=15.0,
            kernel_type="MIXED", confidence="medium", source="estimated"
        ),
        "Softmax": OperatorProfile(
            name="Softmax", category="Activation",
            sm_util=20.0, l2_throughput=45.0, l1_throughput=25.0,
            dram_util=70.0, ipc=0.18, pipeline_util=8.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),
        "Embedding": OperatorProfile(
            name="Embedding", category="Embedding",
            sm_util=5.0, l2_throughput=30.0, l1_throughput=15.0,
            dram_util=90.0, ipc=0.08, pipeline_util=2.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),
        "Dropout": OperatorProfile(
            name="Dropout", category="Other",
            sm_util=5.0, l2_throughput=40.0, l1_throughput=20.0,
            dram_util=85.0, ipc=0.08, pipeline_util=2.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),

        # Pooling
        "AvgPool2d": OperatorProfile(
            name="AvgPool2d", category="Pool",
            sm_util=40.0, l2_throughput=35.0, l1_throughput=20.0,
            dram_util=85.0, ipc=0.4, pipeline_util=18.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),
        "AdaptiveAvgPool2d": OperatorProfile(
            name="AdaptiveAvgPool2d", category="Pool",
            sm_util=35.0, l2_throughput=32.0, l1_throughput=18.0,
            dram_util=80.0, ipc=0.35, pipeline_util=15.0,
            kernel_type="MEMORY_BOUND", confidence="medium", source="estimated"
        ),

        # Other
        "Flatten": OperatorProfile(
            name="Flatten", category="Other",
            sm_util=2.0, l2_throughput=50.0, l1_throughput=30.0,
            dram_util=95.0, ipc=0.05, pipeline_util=1.0,
            kernel_type="MEMORY_BOUND", confidence="low", source="estimated"
        ),
        "Other": OperatorProfile(
            name="Other", category="Other",
            sm_util=10.0, l2_throughput=30.0, l1_throughput=15.0,
            dram_util=50.0, ipc=0.1, pipeline_util=5.0,
            kernel_type="MIXED", confidence="low", source="default"
        ),
    }

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.profiles: Dict[str, OperatorProfile] = {}
        self._lock = threading.RLock()
        self._load_database()

    def _load_database(self):
        """데이터베이스 로드"""
        # 기본 프로파일로 초기화
        self.profiles = dict(self.DEFAULT_PROFILES)

        # 파일에서 추가 로드
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for name, profile_data in data.get("operators", {}).items():
                        self.profiles[name] = OperatorProfile(
                            name=name,
                            category=profile_data.get("category", "Other"),
                            sm_util=profile_data.get("sm_util", 0),
                            l2_throughput=profile_data.get("l2_throughput", 0),
                            l1_throughput=profile_data.get("l1_throughput", 0),
                            dram_util=profile_data.get("dram_util", 0),
                            ipc=profile_data.get("ipc", 0),
                            pipeline_util=profile_data.get("pipeline_util", 0),
                            kernel_type=profile_data.get("kernel_type", "UNKNOWN"),
                            confidence=profile_data.get("confidence", "medium"),
                            source=profile_data.get("source", "database")
                        )
            except Exception as e:
                print(f"[OperatorDB] Warning: Failed to load database: {e}")

    def save_database(self):
        """데이터베이스 저장"""
        with self._lock:
            data = {
                "metadata": {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "operator_count": len(self.profiles)
                },
                "operators": {name: p.to_dict() for name, p in self.profiles.items()}
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)

    def get_profile(self, operator_name: str) -> Optional[OperatorProfile]:
        """오퍼레이터 프로파일 조회"""
        # 정확한 매칭
        if operator_name in self.profiles:
            return self.profiles[operator_name]

        # 패턴 매칭
        normalized = self._normalize_operator_name(operator_name)
        if normalized in self.profiles:
            return self.profiles[normalized]

        # 카테고리 기반 매칭
        category = self._categorize_operator(operator_name)
        if category != "Other":
            for name, profile in self.profiles.items():
                if profile.category == category:
                    return profile

        return self.profiles.get("Other")

    def _normalize_operator_name(self, name: str) -> str:
        """오퍼레이터 이름 정규화"""
        # aten:: 접두사 제거
        name = re.sub(r'^aten::', '', name)

        # 공통 매핑
        mappings = {
            'conv2d': 'Conv2d',
            'conv1d': 'Conv2d',
            'batch_norm': 'BatchNorm2d',
            'layer_norm': 'LayerNorm',
            'linear': 'Linear',
            'matmul': 'Linear',
            'mm': 'Linear',
            'bmm': 'Linear',
            'relu': 'ReLU',
            'relu_': 'ReLU',
            'gelu': 'GELU',
            'silu': 'GELU',
            'max_pool2d': 'MaxPool2d',
            'avg_pool2d': 'AvgPool2d',
            'adaptive_avg_pool2d': 'AdaptiveAvgPool2d',
            'add': 'Add',
            'add_': 'Add',
            'mul': 'Elementwise',
            'mul_': 'Elementwise',
            'cat': 'Elementwise',
            'softmax': 'Softmax',
            'dropout': 'Dropout',
            'embedding': 'Embedding',
            'flatten': 'Flatten',
            '_softmax': 'Softmax',
            'scaled_dot_product_attention': 'Attention',
            'multi_head_attention_forward': 'MultiheadAttention',
        }

        name_lower = name.lower()
        return mappings.get(name_lower, name)

    def _categorize_operator(self, name: str) -> str:
        """오퍼레이터 카테고리 분류"""
        name_lower = name.lower()

        if 'conv' in name_lower:
            return "Conv"
        elif 'linear' in name_lower or 'matmul' in name_lower or name_lower in ['mm', 'bmm']:
            return "Linear"
        elif 'norm' in name_lower:
            return "Norm"
        elif any(act in name_lower for act in ['relu', 'gelu', 'silu', 'softmax', 'sigmoid', 'tanh']):
            return "Activation"
        elif 'pool' in name_lower:
            return "Pool"
        elif 'attention' in name_lower:
            return "Attention"
        elif any(elem in name_lower for elem in ['add', 'mul', 'cat', 'div', 'sub']):
            return "Elementwise"
        elif 'embedding' in name_lower:
            return "Embedding"
        else:
            return "Other"

    def update_profile(self, operator_name: str, profile: OperatorProfile):
        """프로파일 업데이트"""
        with self._lock:
            self.profiles[operator_name] = profile

    def list_operators(self) -> List[str]:
        """등록된 오퍼레이터 목록"""
        return list(self.profiles.keys())


# ==================== Operator Decomposer ====================
class OperatorDecomposer:
    """워크로드를 오퍼레이터 단위로 분해"""

    # Elementwise 패턴 (torch.profiler 출력 기반)
    ELEMENTWISE_PATTERNS = [
        r'aten::add', r'aten::add_', r'aten::mul', r'aten::mul_',
        r'aten::div', r'aten::sub', r'aten::cat', r'aten::clone',
        r'aten::copy_', r'aten::contiguous', r'aten::view', r'aten::reshape',
        r'aten::transpose', r'aten::permute', r'aten::squeeze', r'aten::unsqueeze',
    ]

    def __init__(self, db: OperatorDatabase = None):
        self.db = db or OperatorDatabase()
        self._torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """PyTorch 사용 가능 여부"""
        try:
            import torch
            return True
        except ImportError:
            return False

    def decompose_model(
        self,
        model_name: str,
        batch_size: int = 16,
        input_size: Tuple[int, ...] = None,
        mode: str = "inference",
        num_warmup: int = 3,
        num_runs: int = 10
    ) -> WorkloadProfile:
        """
        모델을 오퍼레이터 단위로 분해

        Args:
            model_name: 모델 이름 (e.g., "resnet50", "bert-base")
            batch_size: 배치 크기
            input_size: 입력 크기 (없으면 모델에 맞게 자동 설정)
            mode: "inference" 또는 "training"
            num_warmup: 워밍업 횟수
            num_runs: 프로파일링 횟수

        Returns:
            WorkloadProfile: 분해된 워크로드 프로파일
        """
        if not self._torch_available:
            return self._create_default_profile(model_name, batch_size, mode)

        import torch
        from torch.profiler import profile, ProfilerActivity

        # 모델 로드
        model, input_data = self._load_model(model_name, batch_size, input_size)
        if model is None:
            return self._create_default_profile(model_name, batch_size, mode)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        elif isinstance(input_data, dict):
            input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in input_data.items()}

        if mode == "inference":
            model.eval()
        else:
            model.train()

        # 워밍업
        with torch.no_grad() if mode == "inference" else torch.enable_grad():
            for _ in range(num_warmup):
                if isinstance(input_data, dict):
                    _ = model(**input_data)
                else:
                    _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        # 프로파일링
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, record_shapes=True) as prof:
            with torch.no_grad() if mode == "inference" else torch.enable_grad():
                for _ in range(num_runs):
                    if isinstance(input_data, dict):
                        output = model(**input_data)
                    else:
                        output = model(input_data)

                    if mode == "training" and hasattr(output, 'loss'):
                        output.loss.backward()
                    elif mode == "training" and isinstance(output, torch.Tensor):
                        output.sum().backward()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

        # 결과 파싱
        return self._parse_profile_results(prof, model_name, batch_size, mode, num_runs)

    def _load_model(self, model_name: str, batch_size: int, input_size: Tuple[int, ...] = None):
        """모델 로드"""
        import torch

        model_name_lower = model_name.lower()

        try:
            # torchvision 모델
            if model_name_lower in ['resnet50', 'resnet18', 'resnet34', 'resnet101']:
                import torchvision.models as models
                model = getattr(models, model_name_lower)(pretrained=False)
                input_data = torch.randn(batch_size, 3, 224, 224)
                return model, input_data

            elif model_name_lower in ['vgg16', 'vgg19']:
                import torchvision.models as models
                model = getattr(models, model_name_lower)(pretrained=False)
                input_data = torch.randn(batch_size, 3, 224, 224)
                return model, input_data

            elif model_name_lower in ['mobilenet_v2', 'mobilenetv2', 'mobilenet']:
                import torchvision.models as models
                model = models.mobilenet_v2(pretrained=False)
                input_data = torch.randn(batch_size, 3, 224, 224)
                return model, input_data

            elif model_name_lower in ['efficientnet_b0', 'efficientnet']:
                import torchvision.models as models
                model = models.efficientnet_b0(pretrained=False)
                input_data = torch.randn(batch_size, 3, 224, 224)
                return model, input_data

            # Transformer 모델 (기본 구현)
            elif 'bert' in model_name_lower or 'transformer' in model_name_lower:
                model = self._create_simple_transformer(
                    hidden_size=768,
                    num_layers=12,
                    num_heads=12,
                    seq_length=128
                )
                input_data = torch.randint(0, 30000, (batch_size, 128))
                return model, input_data

            elif 'gpt' in model_name_lower:
                model = self._create_simple_transformer(
                    hidden_size=768,
                    num_layers=12,
                    num_heads=12,
                    seq_length=128,
                    is_decoder=True
                )
                input_data = torch.randint(0, 30000, (batch_size, 128))
                return model, input_data

            elif 'vit' in model_name_lower:
                import torchvision.models as models
                model = models.vit_b_16(pretrained=False)
                input_data = torch.randn(batch_size, 3, 224, 224)
                return model, input_data

            else:
                print(f"[Decomposer] Unknown model: {model_name}")
                return None, None

        except Exception as e:
            print(f"[Decomposer] Failed to load model {model_name}: {e}")
            return None, None

    def _create_simple_transformer(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        seq_length: int = 128,
        vocab_size: int = 30522,
        is_decoder: bool = False
    ):
        """간단한 Transformer 모델 생성"""
        import torch
        import torch.nn as nn

        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.pos_embedding = nn.Embedding(seq_length, hidden_size)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(hidden_size, vocab_size)

            def forward(self, x):
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                x = self.embedding(x) + self.pos_embedding(positions)
                x = self.transformer(x)
                return self.fc(x)

        return SimpleTransformer()

    def _parse_profile_results(
        self,
        prof,
        model_name: str,
        batch_size: int,
        mode: str,
        num_runs: int
    ) -> WorkloadProfile:
        """프로파일링 결과 파싱"""
        import torch

        operators = {}
        total_time_us = 0

        # 이벤트별 시간 집계
        for event in prof.key_averages():
            name = event.key

            # CUDA 시간 우선, 없으면 CPU 시간
            if hasattr(event, 'self_cuda_time_total') and event.self_cuda_time_total > 0:
                time_us = event.self_cuda_time_total / num_runs
            elif hasattr(event, 'cuda_time_total') and event.cuda_time_total > 0:
                time_us = event.cuda_time_total / num_runs
            elif hasattr(event, 'self_cpu_time_total'):
                time_us = event.self_cpu_time_total / num_runs
            else:
                continue

            if time_us <= 0:
                continue

            # 오퍼레이터 분류
            op_category = self._classify_operator(name)
            if op_category not in operators:
                operators[op_category] = OperatorComposition(
                    name=op_category,
                    category=self.db._categorize_operator(op_category),
                    time_us=0,
                    call_count=0
                )

            operators[op_category].time_us += time_us
            operators[op_category].call_count += event.count // num_runs
            total_time_us += time_us

        # 시간 비율 계산
        for op in operators.values():
            if total_time_us > 0:
                op.time_percent = (op.time_us / total_time_us) * 100
                op.avg_time_us = op.time_us / max(1, op.call_count)

        # 워크로드 프로파일 생성
        workload_id = f"{model_name}_{batch_size}_{mode}"
        profile = WorkloadProfile(
            workload_id=workload_id,
            model_name=model_name,
            batch_size=batch_size,
            mode=mode,
            total_time_ms=total_time_us / 1000,
            operators=operators,
            profiled_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        return profile

    def _classify_operator(self, name: str) -> str:
        """오퍼레이터 이름을 카테고리로 분류"""
        name_lower = name.lower()

        # Elementwise 패턴 체크
        for pattern in self.ELEMENTWISE_PATTERNS:
            if re.search(pattern, name_lower):
                return "Elementwise"

        # 주요 오퍼레이터 매핑
        if 'conv' in name_lower:
            return "Conv2d"
        elif any(x in name_lower for x in ['linear', 'addmm', 'matmul', 'mm', 'bmm']):
            return "Linear"
        elif 'batch_norm' in name_lower:
            return "BatchNorm2d"
        elif 'layer_norm' in name_lower:
            return "LayerNorm"
        elif 'relu' in name_lower:
            return "ReLU"
        elif 'gelu' in name_lower:
            return "GELU"
        elif 'softmax' in name_lower:
            return "Softmax"
        elif 'max_pool' in name_lower:
            return "MaxPool2d"
        elif 'avg_pool' in name_lower or 'adaptive' in name_lower:
            return "AvgPool2d"
        elif 'attention' in name_lower:
            return "Attention"
        elif 'embedding' in name_lower:
            return "Embedding"
        elif 'dropout' in name_lower:
            return "Dropout"
        else:
            return "Other"

    def _create_default_profile(self, model_name: str, batch_size: int, mode: str) -> WorkloadProfile:
        """기본 프로파일 생성 (프로파일링 불가 시)"""
        # 모델 타입별 기본 구성
        model_lower = model_name.lower()

        if any(cnn in model_lower for cnn in ['resnet', 'vgg', 'mobilenet', 'efficientnet']):
            operators = {
                "Conv2d": OperatorComposition(name="Conv2d", category="Conv", time_percent=50.0),
                "Elementwise": OperatorComposition(name="Elementwise", category="Elementwise", time_percent=30.0),
                "Linear": OperatorComposition(name="Linear", category="Linear", time_percent=15.0),
                "Other": OperatorComposition(name="Other", category="Other", time_percent=5.0),
            }
        elif any(trans in model_lower for trans in ['bert', 'gpt', 'transformer', 'vit']):
            operators = {
                "Linear": OperatorComposition(name="Linear", category="Linear", time_percent=60.0),
                "Attention": OperatorComposition(name="Attention", category="Attention", time_percent=15.0),
                "Elementwise": OperatorComposition(name="Elementwise", category="Elementwise", time_percent=15.0),
                "LayerNorm": OperatorComposition(name="LayerNorm", category="Norm", time_percent=5.0),
                "Other": OperatorComposition(name="Other", category="Other", time_percent=5.0),
            }
        else:
            operators = {
                "Other": OperatorComposition(name="Other", category="Other", time_percent=100.0),
            }

        workload_id = f"{model_name}_{batch_size}_{mode}"
        return WorkloadProfile(
            workload_id=workload_id,
            model_name=model_name,
            batch_size=batch_size,
            mode=mode,
            operators=operators,
            profiled_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )


# ==================== Workload Quantifier ====================
class WorkloadQuantifier:
    """워크로드 6D 메트릭 정량화"""

    def __init__(self, db: OperatorDatabase = None):
        self.db = db or OperatorDatabase()

    def quantify(self, workload: WorkloadProfile) -> WorkloadProfile:
        """
        워크로드의 6D 메트릭 계산

        가중 평균 공식:
        Workload_metric = Σ (Operator_metric × time_percent / 100)
        """
        metrics = {
            'sm_util': 0.0,
            'l2_throughput': 0.0,
            'l1_throughput': 0.0,
            'dram_util': 0.0,
            'ipc': 0.0,
            'pipeline_util': 0.0,
        }

        coverage = 0.0

        for op_name, op_comp in workload.operators.items():
            time_weight = op_comp.time_percent / 100.0

            # 오퍼레이터 프로파일 조회
            op_profile = self.db.get_profile(op_name)
            if op_profile:
                coverage += time_weight
                metrics['sm_util'] += op_profile.sm_util * time_weight
                metrics['l2_throughput'] += op_profile.l2_throughput * time_weight
                metrics['l1_throughput'] += op_profile.l1_throughput * time_weight
                metrics['dram_util'] += op_profile.dram_util * time_weight
                metrics['ipc'] += op_profile.ipc * time_weight
                metrics['pipeline_util'] += op_profile.pipeline_util * time_weight

        # 워크로드 프로파일 업데이트
        workload.sm_util = round(metrics['sm_util'], 2)
        workload.l2_throughput = round(metrics['l2_throughput'], 2)
        workload.l1_throughput = round(metrics['l1_throughput'], 2)
        workload.dram_util = round(metrics['dram_util'], 2)
        workload.ipc = round(metrics['ipc'], 3)
        workload.pipeline_util = round(metrics['pipeline_util'], 2)
        workload.coverage = round(coverage * 100, 1)

        # 커널 타입 분류
        workload.kernel_type = self._classify_workload(workload)
        workload.arithmetic_intensity = self._estimate_arithmetic_intensity(workload)

        return workload

    def _classify_workload(self, workload: WorkloadProfile) -> str:
        """워크로드 커널 타입 분류"""
        # SM 활용률과 DRAM 활용률 기반 분류
        sm = workload.sm_util
        dram = workload.dram_util

        if sm > 40 and dram < 60:
            return "COMPUTE_BOUND"
        elif dram > 70 and sm < 30:
            return "MEMORY_BOUND"
        else:
            return "MIXED"

    def _estimate_arithmetic_intensity(self, workload: WorkloadProfile) -> float:
        """
        Arithmetic Intensity 추정

        AI = FLOP / Byte
        높으면 Compute-bound, 낮으면 Memory-bound
        """
        # SM/DRAM 비율 기반 추정
        if workload.dram_util > 0:
            # 간단한 휴리스틱: SM 활용률이 높고 DRAM이 낮으면 AI가 높음
            ai = (workload.sm_util / workload.dram_util) * 50
            return round(min(200, max(0.1, ai)), 1)
        return 0.0


# ==================== Main Interface ====================
class WorkloadProfiler:
    """통합 워크로드 프로파일러 인터페이스"""

    def __init__(self):
        self.db = OperatorDatabase()
        self.decomposer = OperatorDecomposer(self.db)
        self.quantifier = WorkloadQuantifier(self.db)
        self._cache: Dict[str, WorkloadProfile] = {}

    def profile_workload(
        self,
        model_name: str,
        batch_size: int = 16,
        mode: str = "inference",
        use_cache: bool = True
    ) -> WorkloadProfile:
        """
        워크로드 프로파일링 및 정량화

        Returns:
            WorkloadProfile: 6D 메트릭이 포함된 워크로드 프로파일
        """
        cache_key = f"{model_name}_{batch_size}_{mode}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # 1. 오퍼레이터 분해
        profile = self.decomposer.decompose_model(
            model_name=model_name,
            batch_size=batch_size,
            mode=mode
        )

        # 2. 6D 메트릭 정량화
        profile = self.quantifier.quantify(profile)

        # 캐시 저장
        self._cache[cache_key] = profile

        return profile

    def get_scheduling_data(self, profile: WorkloadProfile) -> dict:
        """
        pooling-kernel-scheduling에 전달할 데이터 생성

        Returns:
            dict: 스케줄링에 필요한 데이터
        """
        return {
            "workload_id": profile.workload_id,
            "model_name": profile.model_name,
            "batch_size": profile.batch_size,
            "mode": profile.mode,

            # 6D 메트릭
            "metrics": {
                "sm_util": profile.sm_util,
                "l2_throughput": profile.l2_throughput,
                "l1_throughput": profile.l1_throughput,
                "dram_util": profile.dram_util,
                "ipc": profile.ipc,
                "pipeline_util": profile.pipeline_util,
            },

            # 분류
            "kernel_type": profile.kernel_type,
            "arithmetic_intensity": profile.arithmetic_intensity,

            # 오퍼레이터 구성 (상위 5개)
            "top_operators": self._get_top_operators(profile, 5),

            # 메타데이터
            "coverage": profile.coverage,
            "profiled_at": profile.profiled_at,
        }

    def _get_top_operators(self, profile: WorkloadProfile, n: int) -> List[dict]:
        """상위 n개 오퍼레이터 반환"""
        sorted_ops = sorted(
            profile.operators.values(),
            key=lambda x: x.time_percent,
            reverse=True
        )
        return [
            {"name": op.name, "time_percent": round(op.time_percent, 1)}
            for op in sorted_ops[:n]
        ]

    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()


# ==================== CLI ====================
def main():
    """CLI 테스트"""
    import argparse

    parser = argparse.ArgumentParser(description="Workload Operator Decomposer")
    parser.add_argument("model", nargs="?", default="resnet50", help="Model name")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference")
    parser.add_argument("--list-operators", action="store_true", help="List known operators")
    args = parser.parse_args()

    profiler = WorkloadProfiler()

    if args.list_operators:
        print("\n=== Known Operators ===")
        for name in profiler.db.list_operators():
            op = profiler.db.get_profile(name)
            print(f"  {name}: {op.kernel_type} (SM={op.sm_util}%, DRAM={op.dram_util}%)")
        return

    print(f"\n=== Profiling {args.model} (batch={args.batch}, mode={args.mode}) ===\n")

    profile = profiler.profile_workload(args.model, args.batch, args.mode)

    print(f"Workload: {profile.workload_id}")
    print(f"Total Time: {profile.total_time_ms:.2f} ms")
    print(f"\n--- Operator Composition ---")
    for name, op in sorted(profile.operators.items(), key=lambda x: x[1].time_percent, reverse=True):
        print(f"  {name}: {op.time_percent:.1f}% ({op.time_us:.1f} us)")

    print(f"\n--- 6D Metrics ---")
    print(f"  SM Util:      {profile.sm_util:.1f}%")
    print(f"  L2 Throughput: {profile.l2_throughput:.1f}%")
    print(f"  L1 Throughput: {profile.l1_throughput:.1f}%")
    print(f"  DRAM Util:    {profile.dram_util:.1f}%")
    print(f"  IPC:          {profile.ipc:.3f}")
    print(f"  Pipeline:     {profile.pipeline_util:.1f}%")

    print(f"\n--- Classification ---")
    print(f"  Kernel Type: {profile.kernel_type}")
    print(f"  Arithmetic Intensity: {profile.arithmetic_intensity}")
    print(f"  DB Coverage: {profile.coverage:.1f}%")

    print(f"\n--- Scheduling Data ---")
    sched_data = profiler.get_scheduling_data(profile)
    print(json.dumps(sched_data, indent=2))


if __name__ == "__main__":
    main()
