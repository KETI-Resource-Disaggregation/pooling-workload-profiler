
#!/usr/bin/env python3
import argparse, json, pandas as pd
from datetime import datetime

def parse_ts(s):
    # nvidia-smi timestamp like: 2025/09/19 10:36:01.123
    s = s.replace('/', '-')
    try:
        return datetime.strptime(s.split('.')[0], "%Y-%m-%d %H:%M:%S")
    except:
        # try alt
        return datetime.fromisoformat(s.split('.')[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvsmi_csv", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--kwh_price", type=float, default=120.0)   # KRW/kWh 예시
    ap.add_argument("--gpu_hour_cost", type=float, default=0.0) # 내부 단가가 있으면 입력
    ap.add_argument("--num_gpus", type=int, default=None)       # override if needed
    ap.add_argument("--out_csv", default="summary_tokens_per_kwh.csv")
    args = ap.parse_args()

    with open(args.result_json) as f:
        r = json.load(f)
    start = datetime.fromisoformat(r["start_ts"])
    end   = datetime.fromisoformat(r["end_ts"])
    dur_s = r["duration_s"]
    tokens_per_s = r["tokens_per_s"]

    df = pd.read_csv(args.nvsmi_csv)
    # Expect columns: timestamp,gpu_index,power_w,util_gpu_pct,util_mem_pct,mem_used_mib
    df["ts"] = df["timestamp"].apply(parse_ts)
    win = df[(df["ts"]>=start) & (df["ts"]<=end)].copy()
    if win.empty:
        raise SystemExit("No nvidia-smi rows within start/end range. Check time sync.")
    # cluster power by second (sum over GPUs)
    per_sec = win.groupby("ts")["power_w"].sum()
    power_w_cluster_avg = per_sec.mean()
    kwh_per_job = (power_w_cluster_avg * dur_s) / (1000*3600)
    tokens_per_kwh = (tokens_per_s * 3600) / (power_w_cluster_avg/1000)

    num_gpus = args.num_gpus if args.num_gpus is not None else len(df["gpu_index"].unique())
    gpu_hours = num_gpus * (dur_s/3600.0)
    cost = kwh_per_job*args.kwh_price + gpu_hours*args.gpu_hour_cost

    out = pd.DataFrame([{
        "mode": r.get("mode"),
        "start": r["start_ts"],
        "end": r["end_ts"],
        "duration_s": dur_s,
        "tokens_per_s": tokens_per_s,
        "power_w_cluster_avg": round(power_w_cluster_avg,2),
        "kWh_per_job": round(kwh_per_job,6),
        "Tokens_per_kWh": round(tokens_per_kwh,2),
        "num_gpus": num_gpus,
        "gpu_hours": round(gpu_hours,4),
        "kWh_price": args.kwh_price,
        "gpu_hour_cost": args.gpu_hour_cost,
        "Cost_per_job_KRW": round(cost,2)
    }])
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()

