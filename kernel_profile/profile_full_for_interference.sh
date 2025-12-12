#!/bin/bash
BERT_TRAIN_DIR="/root/kernel_scheduling/mlPerf/training/retired_benchmarks/bert"
BERT_INFER_DIR="/root/kernel_scheduling/mlPerf/inference/language/bert"
PROFILE_DIR="${BERT_TRAIN_DIR}/profiling_interference_vanilla"

mkdir -p ${PROFILE_DIR}
export CUDA_VISIBLE_DEVICES=1

TRAIN_BATCH=32
TRAIN_STEPS=10  # 줄임 (프로파일링 오래 걸려서)
INFER_EXAMPLES=10

echo "=========================================="
echo "Interference Profiling (Vanilla Co-location)"
echo "No MPS - Default CUDA time-slicing"
echo "=========================================="
echo "Start: $(date)"

# Cleanup
pkill -9 -f ncu 2>/dev/null
pkill -9 -f nvidia-cuda-mps 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null
sleep 2

# ===== 1. Training Alone =====
echo ""
echo "[1/3] Training ALONE"
echo "Start: $(date)"

cd ${BERT_TRAIN_DIR}

ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    -o ${PROFILE_DIR}/train_alone \
    python pytorch_bert_train.py \
      --batch_size=${TRAIN_BATCH} \
      --num_steps=${TRAIN_STEPS} \
      --device=cuda:0 \
    > ${PROFILE_DIR}/train_alone.log 2>&1

echo "End: $(date)"
if [ $? -eq 0 ]; then
    echo "✓ Training alone done"
    ls -lh ${PROFILE_DIR}/train_alone.ncu-rep
else
    echo "✗ Failed"
    exit 1
fi

sleep 5

# ===== 2. Inference Alone =====
echo ""
echo "[2/3] Inference ALONE"
echo "Start: $(date)"

cd ${BERT_INFER_DIR}
export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"

ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    -o ${PROFILE_DIR}/infer_alone \
    python3 run.py \
      --backend=pytorch \
      --scenario=Offline \
      --max_examples=${INFER_EXAMPLES} \
    > ${PROFILE_DIR}/infer_alone.log 2>&1

echo "End: $(date)"
if [ $? -eq 0 ]; then
    echo "✓ Inference alone done"
    ls -lh ${PROFILE_DIR}/infer_alone.ncu-rep
else
    echo "✗ Failed"
    exit 1
fi

sleep 5

# ===== 3. Vanilla Co-location (NO MPS) =====
echo ""
echo "[3/3] Vanilla Co-location (NO MPS)"
echo "Start: $(date)"

# Training background
cd ${BERT_TRAIN_DIR}
python pytorch_bert_train.py \
  --batch_size=${TRAIN_BATCH} \
  --num_steps=${TRAIN_STEPS} \
  --device=cuda:0 \
  > ${PROFILE_DIR}/train_coloc.log 2>&1 &
TRAIN_PID=$!

echo "Training PID: ${TRAIN_PID}"
sleep 5

# Inference profiling
cd ${BERT_INFER_DIR}
ncu --target-processes all \
    --replay-mode application \
    --kernel-name-base mangled \
    -o ${PROFILE_DIR}/vanilla_coloc \
    python3 run.py \
      --backend=pytorch \
      --scenario=Offline \
      --max_examples=${INFER_EXAMPLES} \
    > ${PROFILE_DIR}/infer_coloc.log 2>&1

wait ${TRAIN_PID}

echo "End: $(date)"
if [ $? -eq 0 ]; then
    echo "✓ Vanilla co-location done"
    ls -lh ${PROFILE_DIR}/vanilla_coloc.ncu-rep
else
    echo "✗ Failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
ls -lh ${PROFILE_DIR}/*.ncu-rep