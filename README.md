# KRAKEN Workload Profiler

**K**ubernetes **R**esource **A**nalysis and **K**ernel **E**xecution **N**etwork

AI 워크로드의 자원 분해 및 풀링(Resource Disaggregation & Pooling)을 위한 종합 프로파일링 시스템

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🎯 Overview

KRAKEN은 AI/ML 워크로드의 GPU 자원 사용 패턴을 심층 분석하고, 효율적인 자원 분해(Resource Disaggregation) 및 풀링(Resource Pooling) 전략을 수립하기 위한 프로파일링 시스템입니다. 실시간 메트릭 수집, 성능 분석, 그리고 자원 최적화 인사이트를 제공합니다.

### Why KRAKEN?

- 🔍 **Deep Visibility**: GPU compute, memory, interconnect를 포함한 다층적 자원 메트릭 수집
- 📊 **Smart Analysis**: ML 기반 워크로드 특성 분석 및 자동 분류
- 🎯 **Disaggregation-Ready**: 자원 분해 및 풀링 적합성 자동 평가
- ⚡ **Real-time Profiling**: 저오버헤드 실시간 프로파일링
- 🔗 **Kubernetes Native**: K8s 환경과의 seamless 통합

## ✨ Key Features

### Profiling Capabilities

- **Multi-dimensional Metrics**
  - GPU utilization (SM, Tensor Core, Memory Controller)
  - Memory usage patterns (allocation, bandwidth, fragmentation)
  - Kernel execution timeline and dependencies
  - PCIe/NVLink bandwidth and latency
  
- **Workload Characterization**
  - Training vs Inference 자동 분류
  - Compute-bound vs Memory-bound 판별
  - Temporal pattern 및 periodicity 분석
  - Resource contention 감지

- **Disaggregation Analysis**
  - Compute-Memory decoupling 가능성 평가
  - Remote memory access overhead 예측
  - Network sensitivity 분석
  - Cost-performance trade-off 모델링

- **Optimization Insights**
  - Co-location 최적 조합 추천
  - Resource pooling 전략 제안
  - QoS violation 예측 및 경고
  - Hardware utilization 개선 가이드

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   KRAKEN Core Engine                     │
├──────────────────────┬──────────────────────────────────┤
│  Metric Collectors   │     Analysis Pipeline            │
│  ┌─────────────┐    │  ┌────────────────────────┐      │
│  │ GPU Monitor │    │  │ Pattern Detector       │      │
│  │ Memory Prof.│───▶│  │ Interference Modeler   │      │
│  │ Kernel Track│    │  │ Disaggregation Scorer  │      │
│  │ Network Mon.│    │  │ Optimization Engine    │      │
│  └─────────────┘    │  └────────────────────────┘      │
└──────────────────────┴──────────────────────────────────┘
           ↓                          ↓
    ┌─────────────┐           ┌──────────────┐
    │   Storage   │           │ Visualization │
    │   Backend   │           │   Dashboard   │
    │ (TimeSeries)│           │   (Web UI)    │
    └─────────────┘           └──────────────┘
           ↓
    ┌─────────────────────────────┐
    │  Integration Layer          │
    │  - Kubernetes API           │
    │  - SPARK Scheduler          │
    │  - Prometheus/Grafana       │
    └─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
- Python 3.8+
- CUDA 11.0+
- NVIDIA GPU with compute capability 7.0+
- Docker (optional, for containerized deployment)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/keti-ai/kraken-profiler.git
cd kraken-profiler

# Install dependencies
pip install -r requirements.txt

# Install KRAKEN
pip install -e .
```

### Basic Usage

#### 1. Simple Profiling
```python
from kraken import WorkloadProfiler

# Initialize profiler
profiler = WorkloadProfiler(
    workload_name="bert-training",
    sampling_interval=100,  # ms
    metrics=["gpu_util", "memory", "kernel_timeline", "pcie_bw"]
)

# Start profiling
with profiler:
    # Your training/inference code here
    model.train()
    # ...

# Analyze results
report = profiler.get_report()
report.save("profile_report.html")
```

#### 2. Disaggregation Analysis
```python
from kraken import DisaggregationAnalyzer

analyzer = DisaggregationAnalyzer(profile_data="profile_bert.json")

# Evaluate disaggregation potential
result = analyzer.evaluate()
print(f"Disaggregation Score: {result.score:.2f}")
print(f"Recommended Strategy: {result.strategy}")
print(f"Expected Performance Impact: {result.perf_impact:.1%}")
```

#### 3. Co-location Optimization
```python
from kraken import ColocationOptimizer

optimizer = ColocationOptimizer()
optimizer.add_workload("inference_bert", profile="bert_inf.json")
optimizer.add_workload("training_resnet", profile="resnet_train.json")

# Find optimal co-location
recommendation = optimizer.optimize(
    constraints={
        "max_memory_gb": 48,
        "min_qos": 0.95,
        "max_interference": 0.1
    }
)

print(f"Recommended Configuration: {recommendation}")
```

## 📊 Profiling Metrics

### Compute Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| SM Utilization | Streaming Multiprocessor usage | % |
| Tensor Core Util | Tensor Core utilization | % |
| Active Warps | Average active warps per SM | count |
| Kernel Duration | Individual kernel execution time | ms |
| Compute Throughput | Achieved FLOPS | TFLOPS |

### Memory Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| Device Memory Usage | Allocated GPU memory | GB |
| Memory Bandwidth | Actual memory throughput | GB/s |
| Cache Hit Rate | L1/L2 cache efficiency | % |
| Memory Fragmentation | Allocation fragmentation level | % |

### Network Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| PCIe Throughput | Host-Device transfer rate | GB/s |
| NVLink Bandwidth | GPU-GPU communication | GB/s |
| Transfer Latency | Data transfer overhead | μs |

### Application Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| Batch Time | Per-batch processing time | ms |
| Throughput | Samples processed per second | samples/s |
| E2E Latency | End-to-end inference latency | ms |

## 🔍 Analysis Features

### 1. Resource Disaggregation Scoring

KRAKEN은 워크로드의 자원 분해 적합성을 자동으로 평가합니다:
```python
analysis = profiler.analyze_disaggregation_potential()

# Output:
# Disaggregation Analysis Report
# ========================================
# Overall Score: 0.72 (High Potential)
# 
# Breakdown:
#   - Compute-Memory Decoupling: 0.85
#   - Network Sensitivity: 0.65
#   - Temporal Stability: 0.78
# 
# Recommendation: Pool-friendly workload
#   Estimated Performance Impact: -8%
#   Cost Savings Potential: 35%
```

### 2. Interference Prediction

여러 워크로드의 동시 실행 시 간섭 효과를 예측합니다:
```python
predictor = InterferencePredictor()
result = predictor.predict(
    primary="inference_gpt",
    secondary="training_vision",
    metrics=["latency", "throughput"]
)

# Predicted Interference:
#   Primary Impact: +12% latency increase
#   Secondary Impact: -5% throughput decrease
#   Resource Contention: Memory bandwidth (78%)
```

### 3. Temporal Pattern Analysis

워크로드의 시간적 패턴을 분석하여 최적 스케줄링 시점을 제안합니다:
```python
temporal = profiler.analyze_temporal_patterns()

# Detected Patterns:
#   - Type: Periodic (period=1.5s)
#   - Peak Phases: [0.2s-0.8s, 1.7s-2.3s]
#   - Idle Windows: [0.9s-1.6s]
#   
# Co-location Recommendations:
#   - Best Match: Complementary workload with 
#                 inverse temporal pattern
```

## 🎨 Visualization

### Interactive Dashboard
```bash
# Launch web-based dashboard
kraken dashboard --port 8080 --profile profile_data.json
```

Dashboard features:
- Real-time metric streaming
- Historical trend analysis
- Comparative workload view
- Disaggregation heatmaps
- Interactive timeline explorer

### Report Generation
```bash
# Generate comprehensive HTML report
kraken report --input profile_data.json --output report.html

# Export to multiple formats
kraken export --input profile_data.json --format json,csv,parquet
```

## 📈 Use Cases

### 1. GPU Pooling Optimization
```python
# Analyze cluster-wide GPU utilization
cluster_analyzer = ClusterAnalyzer(kubeconfig="~/.kube/config")
utilization = cluster_analyzer.analyze_utilization(timerange="7d")

# Identify pooling opportunities
opportunities = cluster_analyzer.find_pooling_candidates(
    min_savings=0.2,  # 20% cost reduction
    max_perf_impact=0.1  # 10% performance impact
)
```

### 2. Multi-tenancy Planning
```python
# Design optimal multi-tenant configuration
planner = MultiTenancyPlanner()
planner.add_tenants([
    {"name": "team_a", "workloads": [...], "sla": {"p99_latency": 100}},
    {"name": "team_b", "workloads": [...], "sla": {"min_throughput": 1000}}
])

plan = planner.generate_allocation_plan(
    available_gpus=["A100", "A100", "V100"],
    optimization="cost"  # or "performance", "balanced"
)
```

### 3. Cost Optimization
```python
# Find cost-optimal instance types
optimizer = CostOptimizer()
recommendation = optimizer.recommend_instances(
    workload_profile="bert_training.json",
    cloud_provider="aws",  # or "gcp", "azure"
    budget_constraint=5000  # USD/month
)

# Output:
#   Recommended: 3x g5.2xlarge (Spot)
#   Expected Cost: $3,420/month (32% savings)
#   Performance: 95% of on-demand
```

### 4. Performance Troubleshooting
```python
# Identify bottlenecks
troubleshooter = PerformanceTroubleshooter(profile="slow_training.json")
issues = troubleshooter.diagnose()

# Detected Issues:
#   1. Memory Bandwidth Bottleneck (severity: HIGH)
#      - Utilization: 92%
#      - Suggestion: Reduce batch size or use gradient accumulation
#   
#   2. PCIe Transfer Overhead (severity: MEDIUM)
#      - 15% of total time spent on data loading
#      - Suggestion: Enable pinned memory or increase prefetch
```

## 🔧 Advanced Configuration

### Configuration File
```yaml
# kraken_config.yaml
profiler:
  sampling:
    interval_ms: 100
    buffer_size: 10000
    
  metrics:
    gpu:
      enabled: [utilization, temperature, power, clocks]
      detailed_kernels: true
    memory:
      enabled: [allocated, reserved, bandwidth, fragmentation]
      track_allocations: true
    network:
      enabled: [pcie, nvlink]
      measure_latency: true
    
  analysis:
    disaggregation:
      enabled: true
      algorithms: [statistical, ml_based]
      thresholds:
        compute_memory_ratio: 0.3
        network_sensitivity: 0.5
        temporal_variance: 0.2
    
    interference:
      enabled: true
      prediction_model: xgboost
      
  storage:
    backend: timeseries  # or sqlite, postgresql
    retention_days: 30
    compression: true
    
  output:
    formats: [json, html, csv]
    destination: ./profiles
    realtime_streaming: true
```

### Environment Variables
```bash
export KRAKEN_CONFIG=./kraken_config.yaml
export KRAKEN_LOG_LEVEL=INFO
export KRAKEN_STORAGE_PATH=./data
export KRAKEN_ENABLE_GPU_PROFILING=true
```

## 🧪 Benchmark Results

MLPerf 워크로드에 대한 KRAKEN 프로파일링 결과:

| Workload | GPU Util | Mem Util | Disagg Score | Pool Potential | Cost Savings |
|----------|----------|----------|--------------|----------------|--------------|
| BERT-Large Training | 87% | 92% | 0.42 | Medium | 15-20% |
| ResNet-50 Inference | 45% | 38% | 0.78 | High | 40-50% |
| GPT-3 Fine-tuning | 91% | 95% | 0.35 | Low | 5-10% |
| DLRM Training | 72% | 68% | 0.61 | Medium-High | 25-35% |
| Stable Diffusion | 83% | 71% | 0.55 | Medium | 20-30% |

*Results based on NVIDIA A100 40GB GPU*

## 🤝 Integration

### Kubernetes Integration

#### DaemonSet Deployment
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kraken-profiler
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: kraken-profiler
  template:
    metadata:
      labels:
        app: kraken-profiler
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: profiler
        image: keti/kraken-profiler:latest
        securityContext:
          privileged: true
        env:
        - name: KRAKEN_MODE
          value: "daemon"
        - name: KRAKEN_METRICS_PORT
          value: "9090"
        volumeMounts:
        - name: nvidia
          mountPath: /usr/local/nvidia
      volumes:
      - name: nvidia
        hostPath:
          path: /usr/local/nvidia
```

#### Pod Annotation
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
  annotations:
    kraken.keti.ai/profiling: "enabled"
    kraken.keti.ai/profile-interval: "100ms"
    kraken.keti.ai/metrics: "gpu,memory,network"
    kraken.keti.ai/export-format: "json"
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:latest
    # ...
```

### SPARK Scheduler Integration
```python
from kraken.integration import SPARKSchedulerPlugin

# Register KRAKEN with SPARK scheduler
plugin = SPARKSchedulerPlugin(
    profiler_endpoint="http://kraken-api:8080",
    update_interval=60  # seconds
)

# KRAKEN will automatically:
# 1. Profile running workloads
# 2. Send optimization hints to SPARK
# 3. Influence scheduling decisions
```

### Prometheus/Grafana
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kraken'
    static_configs:
      - targets: ['kraken-exporter:9090']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'kraken_.*'
        action: keep
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Profiling Best Practices](docs/best_practices.md)
- [Architecture Deep Dive](docs/architecture.md)
- [Integration Examples](docs/integrations.md)
- [Case Studies](docs/case_studies.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🛠️ Development

### Building from Source
```bash
git clone https://github.com/keti-ai/kraken-profiler.git
cd kraken-profiler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Build package
python setup.py sdist bdist_wheel
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Coverage report
pytest --cov=kraken --cov-report=html
```

## 🗺️ Roadmap

- [ ] **v1.1** (Q1 2025)
  - Multi-GPU profiling support
  - Enhanced ML-based prediction models
  - Kubernetes Operator
  
- [ ] **v1.2** (Q2 2025)
  - AMD GPU support
  - Cloud cost API integration
  - Advanced visualization features
  
- [ ] **v2.0** (Q3 2025)
  - Distributed profiling architecture
  - Real-time optimization engine
  - Multi-cloud resource orchestration

## 📝 Citation

If you use KRAKEN in your research, please cite:
```bibtex
@inproceedings{kraken2025,
  title={KRAKEN: A Comprehensive Workload Profiler for AI Resource Disaggregation},
  author={Kim, Juwon and Team},
  booktitle={Proceedings of the International Conference on AI Systems},
  year={2025},
  organization={KETI}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙋 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Contributors

- Development Team: AI System Research Division, KETI
- Maintainer: Juwon Kim (juwon.kim@keti.re.kr)

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/keti-ai/kraken-profiler/issues)
- **Email**: kraken-support@keti.re.kr
- **Documentation**: https://kraken-profiler.readthedocs.io

## 🏆 Acknowledgments

This project is developed as part of the KETI AI resource management research initiative, supported by the Ministry of Science and ICT, South Korea.

---

<div align="center">

**Korea Electronics Technology Institute (KETI)**

AI System Research Division

*Empowering efficient AI infrastructure through intelligent resource management*

</div>
