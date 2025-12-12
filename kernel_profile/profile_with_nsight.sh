#!/bin/bash
MODE=${1:-vanilla}
BERT_TRAIN_DIR="/root/kernel_scheduling/mlPerf/training/retired_benchmarks/bert"
BERT_INFER_DIR="/root/kernel_scheduling/mlPerf/inference/language/bert"
RESULTS_DIR="${BERT_TRAIN_DIR}/profiling_results/${MODE}"
LIBBLESS_SO="${BERT_TRAIN_DIR}/libbless.so"
SCHEDULER="/root/kernel_scheduling/spark_scheduler.py"

mkdir -p ${RESULTS_DIR}
export CUDA_VISIBLE_DEVICES=0

# 일관된 설정
TRAIN_BATCH=32
TRAIN_STEPS=10
INFER_EXAMPLES=10

echo "=========================================="
echo "Nsight Profiling: ${MODE}"
echo "Training: batch=${TRAIN_BATCH}, steps=${TRAIN_STEPS}"
echo "Inference: examples=${INFER_EXAMPLES}"
echo "WARNING: This will take 30-60 minutes"
echo "=========================================="

# 환경 정리
echo quit | nvidia-cuda-mps-control 2>/dev/null
pkill -f spark_scheduler 2>/dev/null
sleep 1

case $MODE in
  vanilla)
    echo "Profiling VANILLA..."
    
    cd ${BERT_TRAIN_DIR}
    ncu --set full \
        --export ${RESULTS_DIR}/training_profile \
        --force-overwrite \
        --target-processes all \
        python pytorch_bert_train.py --batch_size=${TRAIN_BATCH} --num_steps=${TRAIN_STEPS} --device=cuda:0 &
    TRAIN_PID=$!
    echo "Training profiling PID: ${TRAIN_PID}"
    
    sleep 5
    
    cd ${BERT_INFER_DIR}
    export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"
    ncu --set full \
        --export ${RESULTS_DIR}/inference_profile \
        --force-overwrite \
        --target-processes all \
        python3 run.py --backend=pytorch --scenario=Offline --max_examples=${INFER_EXAMPLES} &
    INFER_PID=$!
    echo "Inference profiling PID: ${INFER_PID}"
    
    wait ${TRAIN_PID}
    echo "Training profiling complete"
    wait ${INFER_PID}
    echo "Inference profiling complete"
    ;;
    
  mps)
    echo "Profiling MPS..."
    
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
    mkdir -p ${CUDA_MPS_PIPE_DIRECTORY} ${CUDA_MPS_LOG_DIRECTORY}
    
    nvidia-cuda-mps-control -d
    sleep 2
    ps aux | grep nvidia-cuda-mps-server | grep -v grep && echo "✓ MPS running"
    
    cd ${BERT_TRAIN_DIR}
    ncu --set full \
        --export ${RESULTS_DIR}/training_profile \
        --force-overwrite \
        --target-processes all \
        python pytorch_bert_train.py --batch_size=${TRAIN_BATCH} --num_steps=${TRAIN_STEPS} --device=cuda:0 &
    TRAIN_PID=$!
    
    sleep 5
    
    cd ${BERT_INFER_DIR}
    export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"
    ncu --set full \
        --export ${RESULTS_DIR}/inference_profile \
        --force-overwrite \
        --target-processes all \
        python3 run.py --backend=pytorch --scenario=Offline --max_examples=${INFER_EXAMPLES} &
    INFER_PID=$!
    
    wait ${TRAIN_PID}
    wait ${INFER_PID}
    
    echo quit | nvidia-cuda-mps-control
    ;;
    
  spark)
    echo "Profiling SPARK..."
    
    # SPARK 환경 설정
    export SQUAD=500
    export SQUAD_TMO_S=2.0
    export BLESS_MASTER=/tmp/bless-master.sock
    
    python3 ${SCHEDULER} > ${RESULTS_DIR}/scheduler.log 2>&1 &
    SCHED_PID=$!
    sleep 2
    
    cd ${BERT_TRAIN_DIR}
    export LD_PRELOAD=${LIBBLESS_SO}
    export BLESS_TENANT=train
    export BLESS_LIMIT_PCT=50
    
    ncu --set full \
        --export ${RESULTS_DIR}/training_profile \
        --force-overwrite \
        --target-processes all \
        python pytorch_bert_train.py --batch_size=${TRAIN_BATCH} --num_steps=${TRAIN_STEPS} --device=cuda:0 &
    TRAIN_PID=$!
    
    sleep 5
    
    cd ${BERT_INFER_DIR}
    export BLESS_TENANT=infer
    export PYTHONPATH="${PWD}/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:${PYTHONPATH}"
    
    ncu --set full \
        --export ${RESULTS_DIR}/inference_profile \
        --force-overwrite \
        --target-processes all \
        python3 run.py --backend=pytorch --scenario=Offline --max_examples=${INFER_EXAMPLES} &
    INFER_PID=$!
    
    wait ${TRAIN_PID}
    wait ${INFER_PID}
    
    kill ${SCHED_PID} 2>/dev/null
    ;;
esac

echo "=========================================="
echo "Profiling Complete: ${MODE}"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
ls -lh ${RESULTS_DIR}/
echo ""
echo "View with:"
echo "  ncu-ui ${RESULTS_DIR}/training_profile.ncu-rep"
echo "  ncu-ui ${RESULTS_DIR}/inference_profile.ncu-rep"
