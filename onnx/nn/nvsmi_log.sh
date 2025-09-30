
#!/usr/bin/env bash
# Log nvidia-smi metrics every 1s to CSV
# Usage: ./nvsmi_log.sh gpu_metrics.csv
OUT=${1:-gpu_metrics.csv}
echo "timestamp,gpu_index,power_w,util_gpu_pct,util_mem_pct,mem_used_mib" > "$OUT"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used \
           --format=csv,noheader -l 1 | while read line; do
  echo "$line" >> "$OUT"
done

