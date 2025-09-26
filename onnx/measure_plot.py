#!/usr/bin/env python3

import argparse, os, csv, json, math

def _abs(p): return os.path.abspath(os.path.expanduser(p))
def _sister(path, new_ext):
    base, _ = os.path.splitext(path)
    return base + new_ext

def _resolve_io(base_or_file: str):
    p = base_or_file
    if p.endswith(".csv"):
        csv_path, json_path = p, _sister(p, ".json")
    elif p.endswith(".json"):
        json_path, csv_path = p, _sister(p, ".csv")
    else:
        csv_path, json_path = p + ".csv", p + ".json"
    base_no_ext = os.path.splitext(os.path.basename(csv_path))[0]
    if not os.path.exists(csv_path) and not os.path.exists(json_path):
        raise SystemExit(f"[ERROR] 입력을 찾을 수 없습니다: '{base_or_file}' (csv/json 모두 없음)")
    if not os.path.exists(csv_path):  csv_path  = None
    if not os.path.exists(json_path): json_path = None
    return csv_path, json_path, base_no_ext

def _to_float(v, default=None):
    try:
        if v is None: return default
        f = float(v)
        if math.isnan(f): return default
        return f
    except Exception:
        return default

def _read_csv(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    t, sm = [], []
    vram_pct = []
    mem_mib, proc_mib = [], []
    mem_total_mib = []

    for row in rows:
        t.append(_to_float(row.get("t_sec")))
        sm.append(_to_float(row.get("sm_pct")))
        vram_pct.append(_to_float(row.get("vram_used_pct")))
        proc_mib.append(_to_float(
            row.get("proc_mem_MiB") or row.get("proc_mem_mib") or row.get("proc_mem")
        ))
        mem_mib.append(_to_float(
            row.get("mem_used_MiB") or row.get("mem_used_mib") or row.get("vram_used_MiB")
        ))
        mem_total_mib.append(_to_float(row.get("mem_total_MiB") or row.get("mem_total_mib")))

    def _compact(tt, vv):
        out_t, out_v = [], []
        for a, b in zip(tt, vv):
            if a is not None and b is not None:
                out_t.append(a); out_v.append(b)
        return out_t, out_v

    t_sm, sm = _compact(t, sm)
    t_vr_pct, vram_pct = _compact(t, vram_pct)
    t_mem_mib, mem_mib = _compact(t, mem_mib)
    t_proc_mib, proc_mib = _compact(t, proc_mib)

    return {
        "t_sm": t_sm, "sm": sm,
        "t_vr_pct": t_vr_pct, "vram_pct": vram_pct,
        "t_mem_mib": t_mem_mib, "mem_mib": mem_mib,
        "t_proc_mib": t_proc_mib, "proc_mib": proc_mib,
        "t_all": t, "mem_total_mib_all": mem_total_mib,
        "samples": len(rows)
    }

def _read_meta(json_path: str):
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r") as f:
            j = json.load(f)
        meta = j.get("meta", {})
        return {
            "mode": meta.get("mode"),
            "model": meta.get("model"),
            "dtype": meta.get("dtype"),
            "batch": meta.get("batch"),
            "seq_in": meta.get("seq_in"),
            "seq_out": meta.get("seq_out"),
            "iters": meta.get("iters"),
            "interval": meta.get("interval_sec"),
            "duration": meta.get("duration_sec"),
            "created_at": meta.get("created_at"),
            "csv_path": meta.get("csv_path")
        }
    except Exception:
        return {}

def _plot_sm(out_png: str, t, sm, title_suffix=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, sm, linewidth=1.2)
    plt.ylim(0, 100)
    plt.xlabel("Time (s)")
    plt.ylabel("SM Utilization (%)")
    ttl = "SM Utilization (%)"
    if title_suffix: ttl += f" – {title_suffix}"
    plt.title(ttl); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def _plot_vram_abs(out_png: str, t, v, unit_label, title_suffix=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, v, linewidth=1.2)
    ymax = max(v) if v else 1.0
    plt.ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(f"VRAM Usage ({unit_label})")
    ttl = f"VRAM Usage ({unit_label})"
    if title_suffix: ttl += f" – {title_suffix}"
    plt.title(ttl); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def _plot_vram_pct(out_png: str, t, pct, title_suffix=""):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, pct, linewidth=1.2)
    plt.ylim(0, 100); plt.xlabel("Time (s)"); plt.ylabel("VRAM Usage (%)")
    ttl = "VRAM Usage (%)"
    if title_suffix: ttl += f" – {title_suffix}"
    plt.title(ttl); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def _derive_vram_abs(data, unit: str):
    if data["t_proc_mib"] and data["proc_mib"]:
        t, v_mib, source = data["t_proc_mib"], data["proc_mib"], "proc"

    elif data["t_mem_mib"] and data["mem_mib"]:
        t, v_mib, source = data["t_mem_mib"], data["mem_mib"], "mem"
    else:
        t, pct = data["t_vr_pct"], data["vram_pct"]
        total_series = data["mem_total_mib_all"]
        if not t or not pct or not total_series:
            label = "MiB" if unit == "mib" else ("GiB" if unit == "gib" else "%")
            return [], [], label, "none"
        total_mib = max([x for x in total_series if x is not None] or [0.0])
        v_mib = [(p/100.0)*total_mib if p is not None else None for p in pct]
        t2, v2 = [], []
        for a, b in zip(t, v_mib):
            if a is not None and b is not None:
                t2.append(a); v2.append(b)
        t, v_mib, source = t2, v2, "pct"

    if unit == "gib":
        return t, [x/1024.0 for x in v_mib], "GiB", source
    elif unit == "mib":
        return t, v_mib, "MiB", source
    else:
        return data["t_vr_pct"], data["vram_pct"], "%", "pct"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV/JSON 경로 또는 확장자 없는 베이스 경로")
    ap.add_argument("--out-dir", default="~/onnx/graph/plots", help="출력 디렉터리 (기본: ~/onnx/graph/plots)")
    ap.add_argument("--vram-unit", dest="vram_unit", choices=["mib","gib","pct"], default="mib",
                    help="VRAM 축 단위 선택 (mib/gib/pct), 기본 mib")
    args = ap.parse_args()

    out_dir = _abs(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for src in args.inputs:
        csv_path, json_path, base = _resolve_io(src)
        if csv_path is None and json_path:
            csv_path = _sister(json_path, ".csv")
        if json_path is None and csv_path:
            json_path = _sister(csv_path, ".json")

        data = _read_csv(csv_path)
        meta = _read_meta(json_path)

        model_str = None
        if meta.get("model"):
            model_str = os.path.basename(str(meta.get("model")))
            if meta.get("dtype"): model_str += f" ({meta.get('dtype')})"
        title_suffix = model_str or base

        out_sm = _abs(os.path.join(out_dir, f"{base}_sm.png"))
        if data["t_sm"] and data["sm"]:
            _plot_sm(out_sm, data["t_sm"], data["sm"], title_suffix=title_suffix)
            print(f"[plot] saved: {out_sm}")
            saved.append(out_sm)
        else:
            print(f"[warn] SM 데이터가 없어 생략: {src}")
            
        if args.vram_unit == "pct":
            if data["t_vr_pct"] and data["vram_pct"]:
                out_vram = _abs(os.path.join(out_dir, f"{base}_vram_pct.png"))
                _plot_vram_pct(out_vram, data["t_vr_pct"], data["vram_pct"], title_suffix=title_suffix)
                print(f"[plot] saved: {out_vram}")
                saved.append(out_vram)
            else:
                print(f"[warn] VRAM(%) 데이터가 없어 생략: {src}")
        else:
            t_abs, v_abs, label, source = _derive_vram_abs(data, args.vram_unit)
            if t_abs and v_abs:
                suffix = "mib" if args.vram_unit == "mib" else "gib"
                out_vram = _abs(os.path.join(out_dir, f"{base}_vram_{suffix}_{source}.png"))
                src_label = {"proc":"Process VRAM", "mem":"Device VRAM", "pct":"VRAM (est.)"}.get(source, "VRAM")
                def _plot_vram_abs_src(out_png, t, v, unit_label, title_suffix=""):
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(11, 4.5))
                    plt.plot(t, v, linewidth=1.2)
                    ymax = max(v) if v else 1.0
                    plt.ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
                    plt.xlabel("Time (s)")
                    plt.ylabel(f"{src_label} ({unit_label})")
                    ttl = f"{src_label} ({unit_label})"
                    if title_suffix: ttl += f" – {title_suffix}"
                    plt.title(ttl); plt.grid(True, alpha=0.3)
                    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
                _plot_vram_abs_src(out_vram, t_abs, v_abs, label, title_suffix=title_suffix)
                print(f"[plot] saved: {out_vram}")
                saved.append(out_vram)
            else:
                print(f"[warn] VRAM({args.vram_unit}) 데이터가 없어 생략: {src}")

    if saved:
        print(f"[done] 총 {len(saved)}개 플롯이 '{out_dir}' 에 저장되었습니다.")
    else:
        print("[done] 저장된 플롯이 없습니다. 입력/CSV 열을 확인해 주세요.")

if __name__ == "__main__":
    main()


