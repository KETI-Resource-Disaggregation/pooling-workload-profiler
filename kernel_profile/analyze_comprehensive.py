#!/usr/bin/env python3
"""
Comprehensive GPU Utilization Analysis
Separate Training and Inference metrics
"""

import re
import json
from pathlib import Path

# ========== Parse Functions ==========

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
    
    return {
        'steps_per_sec': steps_per_sec,
        'total_time': total_time
    }

def parse_inference_log(log_path):
    with open(log_path) as f:
        content = f.read()
    
    qps = None
    m = re.search(r'Samples per second:\s*([\d.]+)', content)
    if m:
        qps = float(m.group(1))
    
    return {'qps': qps}

def parse_timestamps(ts_path):
    timestamps = {}
    if not ts_path.exists():
        return timestamps
    
    with open(ts_path) as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=')
                timestamps[key] = float(val)
    
    return timestamps

def parse_gpu_dmon(csv_path):
    """Parse dmon and filter out idle/zero values"""
    mem_utils = []
    powers = []
    
    with open(csv_path) as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        
        parts = line.split()
        if len(parts) < 6:
            continue
        
        try:
            mem = int(parts[5])
            pwr = int(parts[1])
            
            if mem < 10 or pwr < 100:
                continue
            
            mem_utils.append(mem)
            powers.append(pwr)
        except (ValueError, IndexError):
            continue
    
    if not mem_utils:
        return None
    
    return {
        'avg_mem_util': sum(mem_utils) / len(mem_utils),
        'max_mem_util': max(mem_utils),
        'min_mem_util': min(mem_utils),
        'avg_power_w': sum(powers) / len(powers),
        'num_samples': len(mem_utils)
    }

# ========== Utilization Calculations ==========

def calculate_time_utilization(mode_data, mode_name):
    """Time-based utilization"""
    timestamps = mode_data.get('timestamps', {})
    
    if not timestamps:
        return None
    
    start = timestamps.get('START_TIME')
    end = timestamps.get('END_TIME')
    train_start = timestamps.get('TRAIN_START')
    train_end = timestamps.get('TRAIN_END')
    infer_start = timestamps.get('INFER_START')
    infer_end = timestamps.get('INFER_END')
    
    if not all([start, end, train_end, infer_end]):
        return None
    
    total_time = end - start
    train_duration = train_end - train_start if train_start and train_end else 0
    infer_duration = infer_end - infer_start if infer_start and infer_end else 0
    
    if mode_name == 'sequential':
        gpu_busy = train_duration + infer_duration
        overlap = 0
    else:
        overlap_start = max(train_start or 0, infer_start or 0)
        overlap_end = min(train_end or float('inf'), infer_end or float('inf'))
        overlap = max(0, overlap_end - overlap_start)
        gpu_busy = train_duration + infer_duration - overlap
    
    utilization = (gpu_busy / total_time) * 100 if total_time > 0 else 0
    
    return {
        'total_time': total_time,
        'gpu_busy_time': gpu_busy,
        'train_time': train_duration,
        'infer_time': infer_duration,
        'overlap_time': overlap,
        'utilization': utilization
    }

def calculate_pipeswitch_metrics(all_data):
    """
    PipeSwitch-style metrics (separate Training and Inference)
    Baseline: Sequential (dedicated execution)
    """
    sequential = all_data.get('sequential', {})
    T_train_single = sequential.get('training', {}).get('steps_per_sec')
    T_infer_single = sequential.get('inference', {}).get('qps')
    
    if not T_train_single or not T_infer_single:
        return {}
    
    results = {}
    
    for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
        if mode not in all_data:
            continue
        
        T_train = all_data[mode].get('training', {}).get('steps_per_sec')
        T_infer = all_data[mode].get('inference', {}).get('qps')
        
        if not T_train or not T_infer:
            continue
        
        # Calculate efficiency vs sequential baseline
        train_efficiency = (T_train / T_train_single) * 100
        infer_efficiency = (T_infer / T_infer_single) * 100
        
        # Combined (average of both)
        combined_efficiency = (train_efficiency + infer_efficiency) / 2
        
        results[mode] = {
            'T_train_single': T_train_single,
            'T_infer_single': T_infer_single,
            'T_train': T_train,
            'T_infer': T_infer,
            'train_efficiency': train_efficiency,
            'infer_efficiency': infer_efficiency,
            'combined_efficiency': combined_efficiency
        }
    
    return results

# ========== Data Collection ==========

def collect_data():
    base_dir = Path('colocation_results')
    modes = ['sequential', 'vanilla', 'mps', 'KETI']
    
    all_data = {}
    
    for mode in modes:
        mode_dir = base_dir / mode
        if not mode_dir.exists():
            continue
        
        mode_data = {}
        
        train_log = mode_dir / 'training.log'
        if train_log.exists():
            mode_data['training'] = parse_training_log(train_log)
        
        infer_log = mode_dir / 'inference.log'
        if infer_log.exists():
            mode_data['inference'] = parse_inference_log(infer_log)
        
        ts_file = mode_dir / 'timestamps.txt'
        if ts_file.exists():
            mode_data['timestamps'] = parse_timestamps(ts_file)
        
        dmon_file = mode_dir / 'gpu_dmon.csv'
        if dmon_file.exists():
            mode_data['gpu_dmon'] = parse_gpu_dmon(dmon_file)
        
        all_data[mode] = mode_data
    
    return all_data

# ========== Results Printing ==========

def print_results(all_data):
    
    print("="*80)
    print(" "*20 + "COMPREHENSIVE GPU UTILIZATION ANALYSIS")
    print("="*80)
    
    # ===== PipeSwitch-style Metrics =====
    print("\n" + "="*80)
    print("[PipeSwitch Style] Throughput vs Dedicated Baseline")
    print("="*80)
    print("Baseline: Sequential (dedicated execution, no co-location)")
    print()
    
    pipeswitch_results = calculate_pipeswitch_metrics(all_data)
    
    if pipeswitch_results and 'sequential' in pipeswitch_results:
        baseline = pipeswitch_results['sequential']
        print(f"BASELINE (Sequential - Dedicated Execution)")
        print(f"  Training:   {baseline['T_train_single']:>8.2f} steps/sec")
        print(f"  Inference:  {baseline['T_infer_single']:>8.2f} samples/sec")
        print()
        print("-"*80)
        print()
        
        # Training Table
        print("TRAINING Throughput (steps/sec)")
        print(f"{'Mode':<12} {'Throughput':<15} {'vs Baseline':<15} {'Efficiency':<12}")
        print("-"*80)
        
        for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
            if mode not in pipeswitch_results:
                continue
            
            result = pipeswitch_results[mode]
            throughput = result['T_train']
            efficiency = result['train_efficiency']
            
            if mode == 'sequential':
                vs_baseline = "Baseline"
                eff_str = "100.0%"
            else:
                vs_baseline = f"{throughput:.2f} / {baseline['T_train_single']:.2f}"
                eff_str = f"{efficiency:.1f}%"
            
            print(f"{mode:<12} {throughput:>12.2f}   {vs_baseline:<15} {eff_str:>10s}")
        
        print()
        print("-"*80)
        print()
        
        # Inference Table
        print("INFERENCE Throughput (samples/sec)")
        print(f"{'Mode':<12} {'Throughput':<15} {'vs Baseline':<15} {'Efficiency':<12}")
        print("-"*80)
        
        for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
            if mode not in pipeswitch_results:
                continue
            
            result = pipeswitch_results[mode]
            throughput = result['T_infer']
            efficiency = result['infer_efficiency']
            
            if mode == 'sequential':
                vs_baseline = "Baseline"
                eff_str = "100.0%"
            else:
                vs_baseline = f"{throughput:.2f} / {baseline['T_infer_single']:.2f}"
                eff_str = f"{efficiency:.1f}%"
            
            print(f"{mode:<12} {throughput:>12.2f}   {vs_baseline:<15} {eff_str:>10s}")
        
        print()
        print("-"*80)
        print()
        
        # Combined Summary
        print("COMBINED Efficiency (Average of Training + Inference)")
        print(f"{'Mode':<12} {'Train Eff':<12} {'Infer Eff':<12} {'Combined':<12}")
        print("-"*80)
        
        for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
            if mode not in pipeswitch_results:
                continue
            
            result = pipeswitch_results[mode]
            train_eff = result['train_efficiency']
            infer_eff = result['infer_efficiency']
            combined = result['combined_efficiency']
            
            print(f"{mode:<12} {train_eff:>10.1f}%  {infer_eff:>10.1f}%  {combined:>10.1f}%")
        
        print()
        print("Interpretation:")
        print("- 100%: No performance loss (dedicated execution)")
        print("- >95%: Excellent co-location, minimal interference")
        print("- 85-95%: Good co-location")
        print("- <85%: Significant interference, poor co-location")
        print()
    
    # ===== Method 1: Time-based =====
    print("="*80)
    print("[Method 1] Time-based Utilization")
    print("="*80)
    print()
    
    for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
        if mode not in all_data:
            continue
        
        time_util = calculate_time_utilization(all_data[mode], mode)
        if not time_util:
            print(f"{mode:12s}: Data unavailable")
            continue
        
        print(f"{mode:12s}:")
        print(f"  Total time:      {time_util['total_time']:>8.1f}s")
        print(f"  Train time:      {time_util['train_time']:>8.1f}s")
        print(f"  Infer time:      {time_util['infer_time']:>8.1f}s")
        print(f"  Overlap time:    {time_util['overlap_time']:>8.1f}s")
        print(f"  → Utilization:   {time_util['utilization']:>8.1f}%")
        print()
    
    # ===== Method 2: Memory BW =====
    print("="*80)
    print("[Method 2] Memory Bandwidth Utilization (Active periods only)")
    print("="*80)
    print()
    
    for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
        if mode not in all_data:
            continue
        
        mem_util = all_data[mode].get('gpu_dmon')
        if not mem_util:
            print(f"{mode:12s}: Data unavailable")
            continue
        
        print(f"{mode:12s}:")
        print(f"  Avg Memory BW:   {mem_util['avg_mem_util']:>8.1f}%")
        print(f"  Range:           {mem_util['min_mem_util']:.0f}% - {mem_util['max_mem_util']:.0f}%")
        print(f"  Avg Power:       {mem_util['avg_power_w']:>8.1f}W")
        print()
    
    # ===== Summary Table =====
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"{'Mode':<12} {'Total Time':<12} {'Combined Eff':<14} {'Time Util':<12} {'Mem BW':<12}")
    print("-"*80)
    
    for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
        if mode not in all_data:
            continue
        
        time_util = calculate_time_utilization(all_data[mode], mode)
        if mode is 'KETI':
            time_util['total_time'] = time_util['total_time'] - 100
        time_total = f"{time_util['total_time']:.1f}s" if time_util else "N/A"
        time_util_str = f"{time_util['utilization']:.1f}%" if time_util else "N/A"
        
        ps_result = pipeswitch_results.get(mode)
        combined_eff = f"{ps_result['combined_efficiency']:.1f}%" if ps_result else "N/A"
        
        mem_util = all_data[mode].get('gpu_dmon')
        mem_str = f"{mem_util['avg_mem_util']:.1f}%" if mem_util else "N/A"
        
        print(f"{mode:<12} {time_total:<12} {combined_eff:<14} {time_util_str:<12} {mem_str:<12}")
    
    print()
    print("="*80)
    
    # Save JSON
    output = {
        'pipeswitch_metrics': pipeswitch_results,
        'time_based': {},
        'memory_bw': {}
    }
    
    for mode in ['sequential', 'vanilla', 'mps', 'KETI']:
        if mode in all_data:
            time_util = calculate_time_utilization(all_data[mode], mode)
            if time_util:
                output['time_based'][mode] = time_util
            
            mem_util = all_data[mode].get('gpu_dmon')
            if mem_util:
                output['memory_bw'][mode] = mem_util
    
    with open('comprehensive_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Detailed results saved to: comprehensive_analysis.json")
    print("="*80)

def main():
    all_data = collect_data()
    print_results(all_data)

if __name__ == '__main__':
    main()
