#!/usr/bin/env python3
import argparse, json, os, time, math, datetime as dt
import torch
import torch.nn as nn

# optional stability tweaks
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

def now_ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class TinyBlock(nn.Module):
    def __init__(self, hidden=1024, heads=16):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.qkv = nn.Linear(hidden, hidden*3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.ff1 = nn.Linear(hidden, hidden*4, bias=False)
        self.ff2 = nn.Linear(hidden*4, hidden, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, T, H = x.shape
        qkv = self.qkv(x)                        # (B,T,3H)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = H // self.heads
        q = q.view(B, T, self.heads, head_dim).transpose(1,2)  # (B,heads,T,head_dim)
        k = k.view(B, T, self.heads, head_dim).transpose(1,2)
        v = v.view(B, T, self.heads, head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(head_dim)   # (B,heads,T,T)
        att = self.softmax(att)
        y = att @ v                                            # (B,heads,T,head_dim)
        y = y.transpose(1,2).contiguous().view(B,T,H)          # (B,T,H)
        y = self.proj(y)
        y = self.ff2(torch.relu(self.ff1(y)))
        return y

class TinyModel(nn.Module):
    def __init__(self, layers=6, hidden=1024, heads=16):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        for blk in self.layers:
            x = x + blk(x)
        return self.norm(x)

def run_synthetic(args, device):
    model = TinyModel(layers=args.layers, hidden=args.hidden, heads=args.heads)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(device=device, dtype=dtype)   # <<< ensure weight dtype matches input dtype

    if args.dp and torch.cuda.device_count() > 1 and args.dp_gpus>1:
        model = nn.DataParallel(model, device_ids=list(range(args.dp_gpus)))

    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    # warmup (no grad if inference)
    for _ in range(5):
        if args.train:
            y = model(x); 
            loss = y.mean()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None: p.grad = None
        else:
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    start_ts = now_ts()
    iters = 0
    steps = args.steps
    if args.train:
        for i in range(steps):
            y = model(x)
            loss = (y**2).mean()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None: p.grad = None
            torch.cuda.synchronize()
            iters += 1
    else:
        with torch.no_grad():
            for i in range(steps):
                _ = model(x)
                torch.cuda.synchronize()
                iters += 1

    end = time.time()
    end_ts = now_ts()
    duration = end - start
    tokens = args.batch * args.seq * iters
    tps = tokens / duration
    res = {
        "mode":"synthetic",
        "start_ts": start_ts,
        "end_ts": end_ts,
        "batch": args.batch,
        "seq": args.seq,
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "fp16": args.fp16,
        "train": args.train,
        "dp": args.dp,
        "dp_gpus": args.dp_gpus,
        "iters": iters,
        "duration_s": duration,
        "tokens_total": tokens,
        "tokens_per_s": tps
    }
    if torch.cuda.is_available():
        peak = {f"gpu{i}_peak_mem_MiB": int(torch.cuda.max_memory_allocated(i)/1024/1024) for i in range(torch.cuda.device_count())}
        res.update(peak)
    return res

def run_hf(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    if args.dp and torch.cuda.device_count() > 1 and args.dp_gpus>1:
        model = nn.DataParallel(model, device_ids=list(range(args.dp_gpus)))
    prompt = "Hello, this is a performance test."
    input_ids = tok([prompt]*args.batch, return_tensors="pt", padding=True).input_ids.to(device)
    gen_tokens = args.seq
    # warmup
    _ = model.generate(input_ids, max_new_tokens=16)
    torch.cuda.synchronize()
    start = time.time()
    start_ts = now_ts()
    out = model.generate(input_ids, max_new_tokens=gen_tokens)
    torch.cuda.synchronize()
    end = time.time()
    end_ts = now_ts()
    duration = end - start
    tps = (args.batch * gen_tokens) / duration
    res = {
        "mode":"hf_gpt2",
        "model": model_name,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "batch": args.batch,
        "gen_tokens_per_seq": gen_tokens,
        "duration_s": duration,
        "tokens_total": args.batch*gen_tokens,
        "tokens_per_s": tps,
        "dp": args.dp,
        "dp_gpus": args.dp_gpus,
        "fp16": args.fp16
    }
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic","hf"], default="synthetic")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dp", action="store_true", help="torch.nn.DataParallel")
    ap.add_argument("--dp_gpus", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="mini_infer_result.json")
    args = ap.parse_args()
    torch.cuda.reset_peak_memory_stats()
    if args.mode=="hf":
        res = run_hf(args, args.device)
    else:
        res = run_synthetic(args, args.device)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()

