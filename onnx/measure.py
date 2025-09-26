#!/usr/bin/env python3

__version__ = "1.1.1"

import argparse, os, time, csv, threading, json, re, math, sys
from datetime import datetime

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetClockInfo, nvmlDeviceGetPowerUsage, NVML_CLOCK_SM,
        nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetGraphicsRunningProcesses
    )
except Exception as e:
    raise SystemExit(
        f"[ERROR] pynvml import 실패: {e}\n"
        "  → 설치: pip install nvidia-ml-py3\n"
    )

def _optional_imports():
    mods = {}
    try:
        import torch
        mods["torch"] = torch
    except Exception:
        mods["torch"] = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mods["AutoModelForCausalLM"] = AutoModelForCausalLM
        mods["AutoTokenizer"] = AutoTokenizer
    except Exception:
        mods["AutoModelForCausalLM"] = None
        mods["AutoTokenizer"] = None
    try:
        import onnx, onnxruntime as ort
        mods["onnx"] = onnx
        mods["ort"] = ort
    except Exception:
        mods["onnx"] = None
        mods["ort"] = None
    try:
        import numpy as np
        mods["np"] = np
    except Exception:
        mods["np"] = None
    return mods

MOD = _optional_imports()

def _get_self_proc_mem_mib(handle):
    pid = os.getpid()
    try:
        procs = nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception:
        procs = []
    try:
        gprocs = nvmlDeviceGetGraphicsRunningProcesses(handle)
    except Exception:
        gprocs = []
    used = None
    for p in list(procs) + list(gprocs):
        try:
            if getattr(p, "pid", None) == pid:
                u = getattr(p, "usedGpuMemory", None)
                if u is not None and u >= 0:
                    used = u
                    break
        except Exception:
            pass
    if used is None:
        return -1.0
    return round(float(used) / (1024.0*1024.0), 1)

def _resolve_nvml_index(args):
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        parts = [p.strip() for p in env.split(",") if p.strip()]
        try:
            return int(parts[0])
        except Exception:
            pass
    return int(args.gpu_idx)

def _nvml_preflight(index_int: int):
    try:
        nvmlInit()
        try:
            nvmlDeviceGetHandleByIndex(int(index_int))
            ok = True   
        except Exception:
            ok = False    
        finally:
            nvmlShutdown()
        return ok
    except Exception:
        return False     

def sampler(physical_gpu_idx: int, interval_s: float, out_csv: str, stop_evt: threading.Event):
    nvmlInit()
    try:
        h = nvmlDeviceGetHandleByIndex(physical_gpu_idx)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts","t_sec","gpu",
                "sm_pct","mem_pct",
                "mem_used_MiB","mem_total_MiB","mem_free_MiB","vram_used_pct",
                "proc_mem_MiB",
                "sm_clock_MHz","power_W"
            ])
            t0 = time.time()
            while not stop_evt.is_set():
                ut = nvmlDeviceGetUtilizationRates(h)
                meminfo = nvmlDeviceGetMemoryInfo(h)
                clk = nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM)
                pwr = nvmlDeviceGetPowerUsage(h) / 1000.0
                used_mib = round(meminfo.used / 1024 / 1024, 1)
                total_mib = round(meminfo.total / 1024 / 1024, 1)
                free_mib = round(meminfo.free / 1024 / 1024, 1)
                used_pct = round((meminfo.used / meminfo.total) * 100.0, 1) if meminfo.total else 0.0
                proc_mib = _get_self_proc_mem_mib(h)
                w.writerow([
                    datetime.now().isoformat(timespec="milliseconds"),
                    f"{time.time() - t0:.3f}",
                    physical_gpu_idx,
                    ut.gpu, ut.memory,
                    used_mib, total_mib, free_mib, used_pct,
                    proc_mib,
                    clk, round(pwr, 1)
                ])
                f.flush()
                time.sleep(interval_s)
    finally:
        nvmlShutdown()

def resolve_output_paths(out_arg: str):
    base = out_arg
    if base.endswith(".csv"):
        csv_path = base
        json_path = base[:-4] + "json"
    elif base.endswith(".json"):
        json_path = base
        csv_path = base[:-5] + "csv"
    else:
        csv_path = base + ".csv"
        json_path = base + ".json"
    return os.path.abspath(csv_path), os.path.abspath(json_path)

def _sanitize_model_id(s: str) -> str:
    base = os.path.basename(s)
    if base.lower().endswith(".onnx"):
        base = os.path.splitext(base)[0]
    if "/" in s and not s.lower().endswith(".onnx"):
        base = s.split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-_.")
    return base or "model"

def parse_input_overrides(s):
    m = []
    if not s:
        return m
    for part in (x.strip() for x in s.split(";") if x.strip()):
        bits = part.split(":")
        if len(bits) == 3:
            name, dtype, shape = bits
        elif len(bits) == 2:
            name, shape = bits
            dtype = None
        else:
            raise ValueError(f"input-shape 구문 오류: '{part}' (예: name:dtype:1,128)")
        shp = [int(x) for x in shape.split(",") if x]
        m.append((name, dtype, shp))
    return m

def run_hf_generate(args, csv_path, json_path):
    torch = MOD["torch"]; AutoModelForCausalLM = MOD["AutoModelForCausalLM"]; AutoTokenizer = MOD["AutoTokenizer"]
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise SystemExit("[ERROR] HF 경로는 torch/transformers 설치가 필요합니다. pip install torch transformers")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    assert torch.cuda.is_available(), "CUDA가 사용 불가합니다."
    device = "cuda:0"
    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dmap[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    except Exception as e:
        print(f"[info] SDPA 사용 불가 → eager로 폴백: {e}")
        model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    model.config.pad_token_id = tok.pad_token_id

    batch_prompt = [args.prompt] * args.batch
    enc = tok(batch_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=args.seq_in)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None: attention_mask = attention_mask.to(device)

    stop_evt = threading.Event()
    nvml_idx = _resolve_nvml_index(args)
    if not _nvml_preflight(nvml_idx):
        raise SystemExit(f"[ERROR] NVML sampler index invalid: {nvml_idx}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} / --gpu-idx={args.gpu_idx}.")
    th = threading.Thread(target=sampler, args=(nvml_idx, args.interval, csv_path, stop_evt), daemon=True); th.start()

    with torch.inference_mode():
        _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=min(8, args.seq_out), do_sample=False, pad_token_id=model.config.pad_token_id)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.iters):
            _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.seq_out, do_sample=False, pad_token_id=model.config.pad_token_id)
            torch.cuda.synchronize()
    t1 = time.time()

    stop_evt.set(); th.join()
    _finalize(csv_path, json_path, meta={
        "mode": "hf-generate", "version": __version__,
        "model": args.hf, "dtype": args.dtype, "batch": args.batch,
        "seq_in": args.seq_in, "seq_out": args.seq_out, "iters": args.iters,
        "interval_sec": args.interval, "gpu_physical_index": nvml_idx,
        "duration_sec": round(t1 - t0, 3),
    })

def _dtype_from_str(s, np):
    if s is None: return None
    s = s.lower()
    if s in ("fp32","float32","f32"): return np.float32
    if s in ("fp16","float16","f16"): return np.float16
    if s in ("bf16","bfloat16"): return np.float16
    if s in ("int64","i64"): return np.int64
    if s in ("int32","i32"): return np.int32
    if s in ("int16","i16"): return np.int16
    if s in ("int8","i8"):  return np.int8
    if s in ("uint8","u8"): return np.uint8
    if s in ("bool",): return np.bool_
    return None

def _mk_feed_tensor(inp, overrides, default_batch, default_seq, vocab, np):
    name = inp.name; lname = name.lower()
    dtype_str = None; oshape = list(getattr(inp,"shape",[]) or []); shp = list(oshape)
    for (n,d,s) in overrides:
        if n == name:
            dtype_str = d; shp = s; break
    new_shape = []
    for i, d in enumerate(shp):
        if d in (None, "None"):
            new_shape.append(default_batch if i == 0 else default_seq)
        else:
            try: new_shape.append(int(d))
            except Exception: new_shape.append(default_seq if i>0 else default_batch)
    if "past" in lname and len(new_shape) >= 3:
        idx = None
        for i, d in enumerate(oshape):
            if isinstance(d, str) and ("past" in d.lower() or "seq" in d.lower() or "sequence" in d.lower()):
                idx = i
        if idx is None: idx = len(new_shape) - 2
        if 0 <= idx < len(new_shape): new_shape[idx] = 0
    if len(new_shape) == 0: new_shape = []
    np_dtype = _dtype_from_str(dtype_str, np)
    if np_dtype is None:
        if any(k in lname for k in ("mask","index","ids","position")): np_dtype = np.int64
        else: np_dtype = np.float32

    if "attention_mask" in lname and np_dtype in (np.int64, np.int32, np.int16, np.int8, np.uint8):
        data = np.ones(tuple(new_shape), dtype=np_dtype)
    elif "position_ids" in lname and np_dtype in (np.int64, np.int32):
        if len(new_shape) == 2:
            b, s = new_shape
            row = np.arange(s, dtype=np_dtype)
            data = np.tile(row, (b, 1))
        else:
            data = np.zeros(tuple(new_shape), dtype=np_dtype)
    else:
        if np_dtype in (np.int64, np.int32, np.int16, np.int8, np.uint8):
            high = int(vocab) if vocab and vocab>0 else 1000
            data = np.random.randint(0, high, size=tuple(new_shape), dtype=np_dtype)
        elif np_dtype == np.bool_:
            data = (np.random.randint(0, 2, size=tuple(new_shape)) > 0).astype(np.bool_)
        else:
            data = np.zeros(tuple(new_shape), dtype=np_dtype)
    return name, data

def _finalize(csv_path, json_path, meta):
    import os as _os
    if not _os.path.exists(csv_path):
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}. NVML sampler가 시작되지 않았을 수 있습니다.")
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f); headers = r.fieldnames or []
        for row in r: rows.append(row)
    def _to_float_or_str(v):
        try: return float(v)
        except Exception: return v
    cols = {h: [] for h in (headers or [])}
    for row in rows:
        for h in cols.keys(): cols[h].append(_to_float_or_str(row.get(h)))
    def _safe_max(name):
        if name in cols:
            vals = [x for x in cols[name] if isinstance(x,(int,float)) and not math.isnan(float(x))]
            return max(vals) if vals else None
        return None
    def _safe_avg(name):
        if name in cols:
            vals = [x for x in cols[name] if isinstance(x,(int,float)) and not math.isnan(float(x))]
            return round(sum(vals)/len(vals), 3) if vals else None
        return None
    summary = {
        "peak_vram_MiB": _safe_max("mem_used_MiB"),
        "peak_vram_used_pct": _safe_max("vram_used_pct"),
        "peak_proc_mem_MiB": _safe_max("proc_mem_MiB"),
        "avg_sm_pct": _safe_avg("sm_pct"),
    }
    out_meta = dict(meta)
    out_meta.update({
        "csv_path": _os.path.abspath(csv_path),
        "samples": len(rows),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": "NVML sampled; sm_pct=SM util %, mem_pct=memory controller util %, vram_used_pct=VRAM occupancy %; proc_mem_MiB=-1 if unavailable.",
        "summary": summary,
    })
    with open(json_path, "w") as jf:
        json.dump({"meta": out_meta, "data": cols}, jf, ensure_ascii=False)
    print(f"[done] csv='{_os.path.abspath(csv_path)}', json='{_os.path.abspath(json_path)}'")

def _build_providers(ort, gpu_idx: int, provider: str):
    if provider.upper() == "CUDA":
        base_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "gpu_mem_limit": str(8 * 1024 * 1024 * 1024),
            "do_copy_in_default_stream": "1",
        }
        providers_vis = [("CUDAExecutionProvider", dict(base_opts))]
        providers_phy = [("CUDAExecutionProvider", dict(base_opts, device_id=int(gpu_idx)))]
        return providers_vis, providers_phy
    elif provider.upper() == "CPU":
        return ["CPUExecutionProvider"], ["CPUExecutionProvider"]
    else:
        return [provider], [provider]

def _make_session(args):
    ort = MOD["ort"]
    if ort is None: raise SystemExit("[ERROR] onnxruntime-gpu 가 설치되어야 합니다. pip install onnxruntime-gpu")
    so = ort.SessionOptions()
    if args.opt_level in ("basic","extended","all"):
        try:
            gol = ort.GraphOptimizationLevel
            so.graph_optimization_level = {
                "basic": gol.ORT_ENABLE_BASIC,
                "extended": gol.ORT_ENABLE_EXTENDED,
                "all": gol.ORT_ENABLE_ALL,
            }[args.opt_level]
        except Exception: pass
    providers_vis, providers_phy = _build_providers(ort, args.gpu_idx, args.provider)
    try:
        return ort.InferenceSession(args.onnx, sess_options=so, providers=providers_vis)
    except Exception:
        return ort.InferenceSession(args.onnx, sess_options=so, providers=providers_phy)

def _onnx_introspect_kind(args):
    onnx = MOD["onnx"]
    if onnx is None: return "unknown"
    try:
        m = onnx.load(args.onnx)
    except Exception:
        return "unknown"
    inputs = [i.name for i in m.graph.input]
    outputs = [o.name for o in m.graph.output]
    ops = {n.op_type for n in m.graph.node}
    name_blob = " ".join(inputs + outputs).lower()
    if "input_ids" in name_blob:
        if "present" in name_blob or "past" in name_blob:
            return "ar_decoder"
        return "encoder"
    if "conv" in {op.lower() for op in ops}:
        return "cnn"
    if "attention" in {op.lower() for op in ops} and "present" in name_blob:
        return "ar_decoder"
    return "unknown"

def run_onnx_encoder_or_cnn(args, csv_path, json_path):
    ort = MOD["ort"]; np = MOD["np"]
    if ort is None or np is None:
        raise SystemExit("[ERROR] ONNX 경로는 onnxruntime-gpu, numpy 설치가 필요합니다. pip install onnxruntime-gpu numpy")
    sess = _make_session(args)
    inputs = sess.get_inputs()
    overrides = parse_input_overrides(args.input_shape)
    feed = {}
    for inp in inputs:
        name, arr = _mk_feed_tensor(inp, overrides, args.batch, args.seq_in, args.vocab, np)
        feed[name] = arr
    stop_evt = threading.Event()
    nvml_idx = _resolve_nvml_index(args)
    if not _nvml_preflight(nvml_idx):
        raise SystemExit(f"[ERROR] NVML sampler index invalid: {nvml_idx}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} / --gpu-idx={args.gpu_idx}.")
    th = threading.Thread(target=sampler, args=(nvml_idx, args.interval, csv_path, stop_evt), daemon=True); th.start()
    for _ in range(max(1,args.warmup)): _ = sess.run(None, feed)
    t0 = time.time()
    for _ in range(args.iters): _ = sess.run(None, feed)
    t1 = time.time()
    stop_evt.set(); th.join()
    _finalize(csv_path, json_path, meta={
        "mode": "onnx-1pass", "version": __version__,
        "model": os.path.abspath(args.onnx), "provider": args.provider, "opt_level": args.opt_level,
        "batch": args.batch, "seq_in": args.seq_in, "iters": args.iters,
        "interval_sec": args.interval, "gpu_physical_index": nvml_idx,
        "duration_sec": round(t1 - t0, 3),
    })

def run_onnx_ar_decode(args, csv_path, json_path):
    """Approximate autoregressive decode on ONNX if present/past are available."""
    ort = MOD["ort"]; np = MOD["np"]
    if ort is None or np is None:
        raise SystemExit("[ERROR] ONNX 경로는 onnxruntime-gpu, numpy 설치가 필요합니다. pip install onnxruntime-gpu numpy")
    sess = _make_session(args)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    in_names = [i.name for i in inputs]
    out_names = [o.name for o in outputs]
    overrides = parse_input_overrides(args.input_shape)

    input_ids_name = next((n for n in in_names if "input_ids" in n.lower() or "input"==n.lower()), None)
    attn_mask_name = next((n for n in in_names if "attention_mask" in n.lower()), None)
    pos_ids_name  = next((n for n in in_names if "position_ids" in n.lower()), None)
    past_names = [n for n in in_names if "past" in n.lower()]
    present_names = [n for n in out_names if "present" in n.lower()]

    if not past_names or not present_names:
        print("[warn] present/past 미발견 → 프롬프트 1패스로 폴백")
        return run_onnx_encoder_or_cnn(args, csv_path, json_path)

    feed = {}
    decl = {i.name:i for i in inputs}
    for name in in_names:
        name2, arr = _mk_feed_tensor(decl[name], overrides, args.batch, args.seq_in, args.vocab, np)
        feed[name] = arr

    if input_ids_name:
        feed[input_ids_name] = np.random.randint(0, max(1000, args.vocab), size=(args.batch, args.seq_in), dtype=np.int64)
    if attn_mask_name:
        feed[attn_mask_name] = np.ones((args.batch, args.seq_in), dtype=np.int64)
    if pos_ids_name:
        feed[pos_ids_name] = np.tile(np.arange(args.seq_in, dtype=np.int64), (args.batch,1))

    stop_evt = threading.Event()
    nvml_idx = _resolve_nvml_index(args)
    if not _nvml_preflight(nvml_idx):
        raise SystemExit(f"[ERROR] NVML sampler index invalid: {nvml_idx}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} / --gpu-idx={args.gpu_idx}.")
    th = threading.Thread(target=sampler, args=(nvml_idx, args.interval, csv_path, stop_evt), daemon=True); th.start()

    _ = sess.run(None, feed)

    t0 = time.time()
    for _iter in range(args.iters):
        outs = sess.run(None, feed)
        cache = {on:arr for on,arr in zip(out_names, outs) if "present" in on.lower()}
        past_feed = {k: v for k,v in feed.items() if "past" in k.lower()}
        for step in range(args.seq_out):
            if cache:
                first_present = next(iter(cache.values()))
                for pname in past_names:
                    past_feed[pname] = first_present
            step_feed = dict(past_feed)
            if input_ids_name:
                step_feed[input_ids_name] = np.random.randint(0, max(1000,args.vocab), size=(args.batch,1), dtype=np.int64)
            if attn_mask_name:
                step_feed[attn_mask_name] = np.ones((args.batch, (args.seq_in+step+1)), dtype=np.int64)
            if pos_ids_name:
                step_feed[pos_ids_name] = np.tile(np.arange(args.seq_in+step+1, dtype=np.int64), (args.batch,1))
            outs = sess.run(None, step_feed)
            cache = {on:arr for on,arr in zip(out_names, outs) if "present" in on.lower()}
    t1 = time.time()

    stop_evt.set(); th.join()
    _finalize(csv_path, json_path, meta={
        "mode": "onnx-ar-decode", "version": __version__,
        "model": os.path.abspath(args.onnx), "provider": args.provider, "opt_level": args.opt_level,
        "batch": args.batch, "seq_in": args.seq_in, "seq_out": args.seq_out, "iters": args.iters,
        "interval_sec": args.interval, "gpu_physical_index": nvml_idx,
        "duration_sec": round(t1 - t0, 3),
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-idx", type=int, default=1, help="물리 GPU index (예: GPU1 → 1)")
    ap.add_argument("--onnx", type=str, help=".onnx 모델 경로")
    ap.add_argument("--hf", type=str, help="HF 모델 ID (예: gpt2-medium, bert-base-uncased)")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-in", type=int, default=128)
    ap.add_argument("--seq-out", type=int, default=256, help="AR decode steps / HF generate new tokens")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.1, help="샘플링 주기(초)")
    ap.add_argument("--gpu1-prefix", action="store_true", help="출력 파일명을 항상 gpu1_<model> 형식으로 저장")
    ap.add_argument("--out", type=str, default=None, help="(생략 시 자동) 출력 베이스 이름 또는 경로")
    ap.add_argument("--out-dir", type=str, default="graph", help="출력 디렉터리(기본: graph)")
    ap.add_argument("--prompt", type=str, default="A quick brown fox jumps over the lazy dog.", help="HF 입력 프롬프트")
    ap.add_argument("--provider", type=str, default="CUDA", help="ONNX Runtime EP (CUDA/CPU/…)")
    ap.add_argument("--opt-level", choices=["basic","extended","all"], default="all")
    ap.add_argument("--input-shape", type=str, default=None, help='ONNX 입력 강제 지정. 예) "input_ids:int64:1,128;attention_mask:int64:1,128"')
    ap.add_argument("--vocab", type=int, default=50257, help="정수 입력 생성 시 최대 값")
    ap.add_argument("-V","--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args()

    if args.out:
        out_base = args.out
        if os.path.dirname(out_base) == "":
            out_base = os.path.join(args.out_dir, out_base)
        os.makedirs(os.path.dirname(out_base), exist_ok=True)
    else:
        if args.onnx: model_name = _sanitize_model_id(args.onnx)
        elif args.hf: model_name = _sanitize_model_id(args.hf)
        else: model_name = "model"
        prefix = "gpu1" if args.gpu1_prefix else f"gpu{args.gpu_idx}"
        out_base = f"{prefix}_{model_name}"
        if os.path.dirname(out_base) == "":
            out_base = os.path.join(args.out_dir, out_base)
        os.makedirs(os.path.dirname(out_base), exist_ok=True)

    csv_path, json_path = resolve_output_paths(out_base)

    if args.hf and args.onnx:
        print("[info] --hf/--onnx 동시 지정 → --onnx 우선 사용")
    if args.onnx:
        kind = _onnx_introspect_kind(args)
        print(f"[auto] ONNX model kind = {kind}")
        if kind == "ar_decoder":
            return run_onnx_ar_decode(args, csv_path, json_path)
        else:
            return run_onnx_encoder_or_cnn(args, csv_path, json_path)
    elif args.hf:
        hf_lower = args.hf.lower()
        if any(k in hf_lower for k in ["gpt","llama","mistral","qwen","phi","rwkv","chatglm"]):
            return run_hf_generate(args, csv_path, json_path)
        print("[warn] HF 인코더 모델은 아직 forward-only 모드 미구현 → ONNX 사용 권장")
        return run_hf_generate(args, csv_path, json_path)
    else:
        raise SystemExit("[ERROR] --onnx 또는 --hf 중 하나를 지정하세요.")

if __name__ == "__main__":
    main()



