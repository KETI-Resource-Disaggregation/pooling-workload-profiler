#!/usr/bin/env python3
"""
자원 해지 시간 분석
"""

import statistics
from pathlib import Path

def read_times(filename):
    """Read release times from file"""
    times = []
    filepath = Path(f'resource_release_measurement/{filename}')
    
    if not filepath.exists():
        return []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                times.append(float(line.strip()))
            except:
                continue
    
    return times

def main():
    baseline_times = read_times('baseline_times.txt')
    improved_times = read_times('improved_times.txt')
    
    if not baseline_times or not improved_times:
        print("ERROR: No data found!")
        return
    
    print("="*80)
    print("Resource Release Time Analysis")
    print("="*80)
    print()
    
    # Baseline
    print("[Baseline - 기존 방식]")
    print(f"  Measurements: {baseline_times}")
    print(f"  Average: {statistics.mean(baseline_times):.3f}s")
    print(f"  StdDev:  {statistics.stdev(baseline_times):.3f}s" if len(baseline_times) > 1 else "  StdDev: N/A")
    print()
    
    # Improved
    print("[Improved - KETI 방식 (선제적 해지)]")
    print(f"  Measurements: {improved_times}")
    print(f"  Average: {statistics.mean(improved_times):.3f}s")
    print(f"  StdDev:  {statistics.stdev(improved_times):.3f}s" if len(improved_times) > 1 else "  StdDev: N/A")
    print()
    
    # Reduction
    baseline_avg = statistics.mean(baseline_times)
    improved_avg = statistics.mean(improved_times)
    reduction = baseline_avg - improved_avg
    reduction_pct = (reduction / baseline_avg) * 100
    
    print("="*80)
    print("RESULT")
    print("="*80)
    print(f"Baseline release time:    {baseline_avg:.3f}s")
    print(f"Improved release time:    {improved_avg:.3f}s")
    print(f"Time reduction:           {reduction:.3f}s")
    print(f"Reduction rate:           {reduction_pct:.1f}%")
    print()
    
    # Target evaluation (PDF 목표: 3~6초 → <1초)
    target_baseline = 4.5  # 평균 3~6초
    target_improved = 1.0  # 목표 <1초
    target_reduction = ((target_baseline - target_improved) / target_baseline) * 100
    
    print("Target (from PDF):")
    print(f"  Expected baseline:  ~{target_baseline}s (3~6s)")
    print(f"  Expected improved:  <{target_improved}s")
    print(f"  Expected reduction: ~{target_reduction:.1f}%")
    print()
    
    if reduction_pct >= target_reduction:
        print("✓ Target ACHIEVED!")
    else:
        print("✗ Target NOT achieved")
    
    print("="*80)

if __name__ == '__main__':
    main()
