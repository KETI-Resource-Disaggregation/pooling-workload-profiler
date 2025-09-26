
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')


def _sanitize_model_id(model_path_or_id: str) -> str:
    if not model_path_or_id:
        return "model"
    base = os.path.basename(model_path_or_id)
    if "." in base:
        base = os.path.splitext(base)[0]
    base = (base.strip() or "model").replace(" ", "-")
    return base


def _bytes_to_mib(b) -> float:
    return float(b) / (1024.0 ** 2)


def _get_est_activation_peak_bytes(summary: dict):
    if "estimated_activation_peak_bytes" in summary:
        return float(summary["estimated_activation_peak_bytes"])
    if "activation_memory_peak_bytes" in summary:
        return float(summary["activation_memory_peak_bytes"])
    tot = summary.get("estimated_total_memory_peak_bytes")
    prm = summary.get("parameter_memory_bytes")
    if isinstance(tot, (int, float)) and isinstance(prm, (int, float)):
        return float(tot) - float(prm)
    return None


def analyze_sm_utilization(df_timeline: pd.DataFrame, model_id: str, out_dir: str):

    col = "sm_util_window_rel_to_peak"
    if col not in df_timeline.columns:
        raise KeyError(f"timeline에 '{col}' 컬럼이 없습니다.")

    util = df_timeline[col].astype(float)

    peak_idx = util.idxmax()
    peak_step = int(df_timeline.loc[peak_idx, "step"])
    peak_pct = float(util.loc[peak_idx] * 100.0)
    print(f"  - Original Predicted Peak SM Utilization: {peak_pct:.2f}% @ step {peak_step}")

    plt.figure(figsize=(16, 7))
    plt.plot(df_timeline["step"], util * 100.0, linewidth=2, label="Predicted SM Util (%)")
    plt.scatter([peak_step], [peak_pct], s=100, zorder=5, label=f"Peak ({peak_pct:.2f}%)")

    plt.title(f"Predicted SM Utilization - {model_id}", fontsize=16)
    plt.xlabel("Operation Step")
    plt.ylabel("Predicted SM Utilization (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()

    out_path = os.path.abspath(os.path.join(out_dir, f"estimate_{model_id}_sm.png"))
    plt.savefig(out_path, dpi=300)
    print(f"  ✅ SM Utilization graph saved to '{out_path}'")
    plt.close()

def analyze_vram(df_timeline: pd.DataFrame, summary: dict, model_id: str, out_dir: str):

    col = "activation_live_bytes_after"
    if col not in df_timeline.columns:
        raise KeyError(f"timeline에 '{col}' 컬럼이 없습니다.")

    mo = summary.get("memory_overhead") or {}
    F = float(mo.get("applied_factor_F", 1.0))

    baseline_peak = summary.get("activation_memory_peak_baseline_bytes") 
    target_peak = _get_est_activation_peak_bytes(summary)             

    raw_bytes = df_timeline[col].astype(float)
    raw_peak = float(raw_bytes.max())

    already_scaled = False
    if isinstance(baseline_peak, (int, float)) and baseline_peak > 0:
        expect_after_F = float(baseline_peak) * F
        relerr = abs(raw_peak - expect_after_F) / max(expect_after_F, 1.0)
        already_scaled = (relerr < 0.25) 

    series_bytes = raw_bytes if already_scaled else (raw_bytes * F)

    if isinstance(target_peak, (int, float)) and target_peak > 0:
        base_peak = float(series_bytes.max())
        if base_peak > 0:
            ratio = float(target_peak) / base_peak
            series_bytes = series_bytes * ratio
        else:
            print("  - Skip alignment (base_peak=0)")
    else:
        print("  - No target peak in summary; use selected series as is.")


    series_mib = series_bytes.apply(_bytes_to_mib)
    peak_val = float(series_mib.max())
    peak_step = int(df_timeline.loc[series_mib.idxmax(), "step"])

    status = "already-scaled" if already_scaled else "scaled-now"
    print(f"\n - Selected peak (plotted): {peak_val:.2f} MiB @ step {peak_step}")

    plt.figure(figsize=(16, 7))
    plt.plot(df_timeline["step"], series_mib, linewidth=2, label="Activation VRAM (MiB)")
    plt.scatter([peak_step], [peak_val], s=100, zorder=5, label=f"Peak ({peak_val:.2f} MiB)")

    plt.title(f"Estimated Activation VRAM per Step - {model_id}", fontsize=16)
    plt.xlabel("Operation Step")
    plt.ylabel("Activation VRAM (MiB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.tight_layout()

    out_path = os.path.abspath(os.path.join(out_dir, f"estimate_{model_id}_vram.png"))
    plt.savefig(out_path, dpi=300)
    print(f"  ✅ VRAM graph saved to '{out_path}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Static analysis plots: SM & Activation VRAM (MiB, single curve)"
    )
    parser.add_argument('--timeline', type=str, required=True, help='Path to the timeline.json file.')
    parser.add_argument('--deps', type=str, required=True, help='Path to the deps.json file (reserved).')
    parser.add_argument('--summary', type=str, required=True, help='Path to the summary.json file.')
    parser.add_argument('--out-dir', type=str, default='estimate', help='출력 디렉터리 (기본: estimate)')
    args = parser.parse_args()

    print("\n--- Starting Static Model Analysis ---")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] Output directory: {os.path.abspath(args.out_dir)}")

    with open(args.timeline, 'r', encoding='utf-8') as f:
        timeline_data = json.load(f)
    with open(args.summary, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    df_timeline = pd.DataFrame(timeline_data['timeline'])

    model_name = timeline_data.get('model_path', '') or summary_data.get('model_name', '')
    model_id = _sanitize_model_id(model_name)

    analyze_sm_utilization(df_timeline, model_id, args.out_dir)

    analyze_vram(df_timeline, summary_data, model_id, args.out_dir)

    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    main()



