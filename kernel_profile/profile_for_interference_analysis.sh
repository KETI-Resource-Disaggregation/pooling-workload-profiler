#!/bin/bash
BERT_TRAIN_DIR="/root/kernel_scheduling/mlPerf/training/retired_benchmarks/bert"
BERT_INFER_DIR="/root/kernel_scheduling/mlPerf/inference/language/bert"
PROFILE_DIR="${BERT_TRAIN_DIR}/profiling_interference"

mkdir -p ${PROFILE_DIR}
export CUDA_VISIBLE_DEVICES=1

TRAIN_BATCH=32
TRAIN_STEPS=20
INFER_EXAMPLES=10

echo "=========================================="
echo "Profiling for Interference Analysis (FIXED)"
echo "=========================================="

# Clean up
echo quit | nvidia-cuda-mps-control 2>/dev/null
pkill -9 nvidia-cuda-mps-server 2>/dev/null
sleep 2

# ===== 1. Training 단독 =====
echo ""
echo "[1/3] Profiling Training ALONE..."
cd ${BERT_TRAIN_DIR}

ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    --launch-skip 5 \
    --launch-count 100 \
    -o ${PROFILE_DIR}/train_alone \
    python pytorch_bert_train.py \
      --batch_size=${TRAIN_BATCH} \
      --num_steps=${TRAIN_STEPS} \
      --device=cuda:0 \
    > ${PROFILE_DIR}/train_alone.log 2>&1

echo "✓ Training alone done"
sleep 5

# ===== 2. Inference 단독 =====
echo ""
echo "[2/3] Profiling Inference ALONE..."
cd ${BERT_INFER_DIR}
export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"

ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    --launch-skip 5 \
    --launch-count 100 \
    -o ${PROFILE_DIR}/infer_alone \
    python3 run.py \
      --backend=pytorch \
      --scenario=Offline \
      --max_examples=${INFER_EXAMPLES} \
    > ${PROFILE_DIR}/infer_alone.log 2>&1

echo "✓ Inference alone done"
sleep 5

# ===== 3. Co-location (MPS) =====
echo ""
echo "[3/3] Profiling Co-location..."

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p ${CUDA_MPS_PIPE_DIRECTORY} ${CUDA_MPS_LOG_DIRECTORY}
nvidia-cuda-mps-control -d
sleep 2

# Training 백그라운드
cd ${BERT_TRAIN_DIR}
python pytorch_bert_train.py \
  --batch_size=${TRAIN_BATCH} \
  --num_steps=${TRAIN_STEPS} \
  --device=cuda:0 \
  > ${PROFILE_DIR}/train_coloc.log 2>&1 &
TRAIN_PID=$!

sleep 3

# Inference 프로파일
cd ${BERT_INFER_DIR}
ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    --launch-skip 5 \
    --launch-count 100 \
    -o ${PROFILE_DIR}/coloc \
    python3 run.py \
      --backend=pytorch \
      --scenario=Offline \
      --max_examples=${INFER_EXAMPLES} \
    > ${PROFILE_DIR}/infer_coloc.log 2>&1

wait ${TRAIN_PID}

echo quit | nvidia-cuda-mps-control
sleep 2

echo ""
echo "=========================================="
echo "✓ ALL PROFILING COMPLETE!"
echo "=========================================="
ls -lh ${PROFILE_DIR}/*.ncu-rep
