#!/usr/bin/env python3
import re
import json
from pathlib import Path

def parse_training_log(log_path):
    with open(log_path) as f:
        content = f.read()
    
    steps_per_sec = None
    m = re.search(r'Steps/sec:\s*([\d.]+)', content)
    if m:
        steps_per_sec = float(m.group(1))
    
    total_time = None
    m = re.search(r'Total time:\s*([\d.]+)s', content)
    if m:
        total_time = float(m.group(1))
    
    final_loss = None
    m = re.search(r'Final loss:\s*([\d.]+)', content)
    if m:
        final_loss = float(m.group(1))
    
    return {
        'steps_per_sec': steps_per_sec,
        'total_time': total_time,
        'final_loss': final_loss
    }

def parse_inference_log(log_path):
    with open(log_path) as f:
        content = f.read()
    
    qps = None
    m = re.search(r'Samples per second:\s*([\d.]+)', content)
    if m:
        qps = float(m.group(1))
    
    lat_p90 = None
    m = re.search(r'Latency P90:\s*([\d.]+)', content)
    if m:
        lat_p90 = float(m.group(1))
    
    lat_mean = None
    m = re.search(r'Mean latency:\s*([\d.]+)', content)
    if m:
        lat_mean = float(m.group(1))
    
    return {
        'qps': qps,
        'lat_p90_ms': lat_p90,
        'lat_mean_ms': lat_mean
    }

def parse_gpu_dmon(csv_path):
    """Parse nvidia-smi dmon output - REAL utilization"""
    sm_utils = []
    mem_utils = []
    powers = []
    
    with open(csv_path) as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # Skip header lines starting with #
        if line.startswith('#') or not line:
            continue
        
        parts = line.split()
        if len(parts) < 9:
            continue
        
        try:
            # Format: gpu pwr gtemp mtemp sm mem enc dec mclk pclk
            sm = int(parts[4])      # SM utilization
            mem = int(parts[5])     # Memory utilization
            pwr = int(parts[1])     # Power
            
            sm_utils.append(sm)
            mem_utils.append(mem)
            powers.append(pwr)
        except (ValueError, IndexError):
            continue
    
    if not sm_utils:
        return None
    
    return {
        'avg_sm_util': sum(sm_utils) / len(sm_utils),
        'max_sm_util': max(sm_utils),
        'avg_mem_util': sum(mem_utils) / len(mem_utils),
        'max_mem_util': max(mem_utils),
        'avg_power_w': sum(powers) / len(powers)
    }

def main():
    base_dir = Path('colocation_results')
    modes = ['vanilla', 'mps', 'spark']
    
    results = {}
    
    for mode in modes:
        mode_dir = base_dir / mode
        if not mode_dir.exists():
            print(f"Warning: {mode_dir} not found, skipping")
            continue
        
        train_log = mode_dir / 'training.log'
        infer_log = mode_dir / 'inference.log'
        gpu_dmon = mode_dir / 'gpu_dmon.csv'
        
        result = {'mode': mode}
        
        if train_log.exists():
            result['training'] = parse_training_log(train_log)
        
        if infer_log.exists():
            result['inference'] = parse_inference_log(infer_log)
        
        if gpu_dmon.exists():
            gpu_data = parse_gpu_dmon(gpu_dmon)
            if gpu_data:
                result['gpu'] = gpu_data
        
        results[mode] = result
    
    # Save JSON
    with open('benchmark_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    print("MLPerf BERT Co-location Benchmark Results")
    print("="*70)
    
    # Training throughput
    print("\n[Training Throughput (steps/sec)]")
    vanilla_train = results.get('vanilla', {}).get('training', {}).get('steps_per_sec')
    for mode in modes:
        if mode not in results:
            continue
        sps = results[mode].get('training', {}).get('steps_per_sec')
        if sps and vanilla_train:
            improvement = ((sps / vanilla_train) - 1) * 100
            print(f"  {mode:10s}: {sps:6.2f}  ({improvement:+.1f}% vs vanilla)")
    
    # Inference throughput
    print("\n[Inference Throughput (QPS)]")
    vanilla_infer = results.get('vanilla', {}).get('inference', {}).get('qps')
    for mode in modes:
        if mode not in results:
            continue
        qps = results[mode].get('inference', {}).get('qps')
        if qps and vanilla_infer:
            improvement = ((qps / vanilla_infer) - 1) * 100
            print(f"  {mode:10s}: {qps:6.2f}  ({improvement:+.1f}% vs vanilla)")
    
    # REAL GPU utilization
    print("\n[Average SM Utilization (%) - REAL]")
    for mode in modes:
        if mode not in results:
            continue
        sm_util = results[mode].get('gpu', {}).get('avg_sm_util')
        mem_util = results[mode].get('gpu', {}).get('avg_mem_util')
        if sm_util is not None:
            print(f"  {mode:10s}: SM={sm_util:5.1f}%  Memory={mem_util:5.1f}%")
    
    # Combined score
    print("\n[Combined Score (Training + Inference, normalized)]")
    vanilla_combined = 1.0
    if vanilla_train and vanilla_infer:
        for mode in modes:
            if mode not in results:
                continue
            sps = results[mode].get('training', {}).get('steps_per_sec', 0)
            qps = results[mode].get('inference', {}).get('qps', 0)
            if sps and qps:
                combined = (sps / vanilla_train) * 0.5 + (qps / vanilla_infer) * 0.5
                improvement = (combined - 1) * 100
                print(f"  {mode:10s}: {combined:.3f}  ({improvement:+.1f}% vs vanilla)")

if __name__ == '__main__':
    main()
