#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
BERT_TRAIN_DIR="/root/kernel_scheduling/mlPerf/training/retired_benchmarks/bert"
BERT_INFER_DIR="/root/kernel_scheduling/mlPerf/inference/language/bert"
RESULTS_DIR="${BERT_TRAIN_DIR}/profiling_results/vanilla_coloc_gpu1"

mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "Vanilla Co-location (GPU 1)"
echo "Training: 20 steps | Inference: 10 examples"
echo "=========================================="

# Training 백그라운드
cd ${BERT_TRAIN_DIR}
python pytorch_bert_train.py \
  --batch_size=16 \
  --num_steps=20 \
  --device=cuda:0 > ${RESULTS_DIR}/training.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: ${TRAIN_PID}"

# 3초 대기
sleep 3

# Inference 시작
cd ${BERT_INFER_DIR}
export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"
python3 run.py \
  --backend=pytorch \
  --scenario=Offline \
  --max_examples=10 > ${RESULTS_DIR}/inference.log 2>&1 &
INFER_PID=$!
echo "Inference PID: ${INFER_PID}"

# GPU 모니터링
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
  --format=csv -l 1 > ${RESULTS_DIR}/gpu_monitor.csv &
MONITOR_PID=$!

echo "Waiting for completion..."
wait ${TRAIN_PID}
echo "Training done"
wait ${INFER_PID}
echo "Inference done"

kill ${MONITOR_PID} 2>/dev/null
echo "Workloads complete!"
