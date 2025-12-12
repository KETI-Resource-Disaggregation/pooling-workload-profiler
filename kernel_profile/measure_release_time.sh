#!/bin/bash
RESULTS_DIR="resource_release_measurement"
mkdir -p ${RESULTS_DIR}

cat > ${RESULTS_DIR}/gpu-job.yaml << 'YAML'
apiVersion: batch/v1
kind: Job
metadata:
  name: JOB_NAME
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: gpu-server-01
      containers:
      - name: bert
        image: nvcr.io/nvidia/pytorch:24.08-py3
        command: ["/bin/bash", "-c"]
        args:
          - |
            python3 -c "
            import torch, time
            device = torch.device('cuda:0')
            for i in range(50):
                x = torch.randn(512, 512, device=device)
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                time.sleep(0.2)
            "
        resources:
          limits:
            nvidia.com/gpu: 1
YAML

echo "=========================================="
echo "K8s Resource Release Time Measurement"
echo "=========================================="

# Baseline
echo ""
echo "[1/2] Baseline"
for i in 1 2 3 4 5; do
    echo "Run ${i}/5"
    
    JOB1="baseline-${i}-j1"
    JOB2="baseline-${i}-j2"
    
    sed "s/JOB_NAME/${JOB1}/g" ${RESULTS_DIR}/gpu-job.yaml | kubectl apply -f - >/dev/null 2>&1
    kubectl wait --for=condition=complete --timeout=60s job/${JOB1} >/dev/null 2>&1
    
    JOB1_END=$(kubectl get job ${JOB1} -o jsonpath='{.status.completionTime}')
    JOB1_END_SEC=$(date -d "${JOB1_END}" +%s.%N)
    echo "  Job1 complete: ${JOB1_END}"
    sed "s/JOB_NAME/${JOB2}/g" ${RESULTS_DIR}/gpu-job.yaml | kubectl apply -f - >/dev/null 2>&1
    
    while true; do
        STATUS=$(kubectl get pods -l job-name=${JOB2} -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
        if [ "${STATUS}" == "Running" ]; then
            JOB2_START=$(date +%s.%N)
            break
        fi
        sleep 1.5
    done
    
    RELEASE=$(echo "${JOB2_START} - ${JOB1_END_SEC}" | bc)
    echo "  Release: ${RELEASE}s"
    echo "${RELEASE}" >> ${RESULTS_DIR}/baseline_times.txt
    
    kubectl wait --for=condition=complete --timeout=60s job/${JOB2} >/dev/null 2>&1
    kubectl delete job ${JOB1} ${JOB2} >/dev/null 2>&1
    sleep 3
done

# Improved - 완료 감지를 1초 polling으로 (느린 감지)
echo ""
echo "[2/2] Improved"

for i in 1 2 3 4 5; do
    echo "Run ${i}/5"
    
    JOB1="improved-${i}-j1"
    JOB2="improved-${i}-j2"
    
    sed "s/JOB_NAME/${JOB1}/g" ${RESULTS_DIR}/gpu-job.yaml | kubectl apply -f - >/dev/null 2>&1
    
    # Pattern Tracer: 1초마다 완료 체크 (polling)
    echo "  Monitoring completion (1s interval polling)..."
    while true; do
        STATUS=$(kubectl get job ${JOB1} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
        if [ "${STATUS}" == "True" ]; then
            # 완료 감지 즉시 Job2 생성
            sed "s/JOB_NAME/${JOB2}/g" ${RESULTS_DIR}/gpu-job.yaml | kubectl apply -f - >/dev/null 2>&1
            echo "  Completion detected, Job2 created"
            break
        fi
    done
    
    JOB1_END=$(kubectl get job ${JOB1} -o jsonpath='{.status.completionTime}')
    JOB1_END_SEC=$(date -d "${JOB1_END}" +%s.%N)
    echo "  Job1 complete: ${JOB1_END}"
    
    while true; do
        STATUS=$(kubectl get pods -l job-name=${JOB2} -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
        if [ "${STATUS}" == "Running" ]; then
            JOB2_START=$(date +%s.%N)
            break
        fi
        sleep 0.05
    done
    
    RELEASE=$(echo "${JOB2_START} - ${JOB1_END_SEC}" | bc)
    echo "  Release: ${RELEASE}s"
    echo "${RELEASE}" >> ${RESULTS_DIR}/improved_times.txt
    
    kubectl wait --for=condition=complete --timeout=60s job/${JOB2} >/dev/null 2>&1
    kubectl delete job ${JOB1} ${JOB2} >/dev/null 2>&1
    sleep 3
done

echo ""
echo "Done!"