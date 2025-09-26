#!/usr/bin/env python3

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import onnx

def _stem(p: Path) -> str:
    return p.name if p.is_dir() else p.stem

def load_onnx(path: str):
    return onnx.load(path)

def build_prod_cons_maps(model) -> Tuple[Dict[str,int], Dict[int,List[int]], Dict[int,List[str]]]:
    graph = model.graph
    producer_of: Dict[str,int] = {}
    outputs_of_node: Dict[int,List[str]] = {}

    for i, node in enumerate(graph.node):
        outs = [o for o in node.output if o]
        outputs_of_node[i] = outs
        for o in outs:
            producer_of[o] = i

    consumers_of_node: Dict[int,List[int]] = {i: [] for i,_ in enumerate(graph.node)}
    for j, node in enumerate(graph.node):
        for x in node.input:
            if x and x in producer_of:
                prod = producer_of[x]
                consumers_of_node[prod].append(j)

    return producer_of, consumers_of_node, outputs_of_node

def node_io_lists(model, idx: int) -> Tuple[List[str], List[str]]:
    n = model.graph.node[idx]
    ins = [x for x in n.input if x]
    outs = [x for x in n.output if x]
    return ins, outs

def compute_levels(model) -> Tuple[List[List[int]], List[int], List[int]]:

    producer_of, consumers_of_node, _ = build_prod_cons_maps(model)
    N = len(model.graph.node)

    indeg_from_nodes = [0]*N
    for i in range(N):
        ins, _ = node_io_lists(model, i)
        cnt = 0
        for x in ins:
            if x in producer_of:
                cnt += 1
        indeg_from_nodes[i] = cnt
        
    ready = [i for i in range(N) if indeg_from_nodes[i]==0]
    levels: List[List[int]] = []
    level_of = [-1]*N
    k = 0
    while ready:
        levels.append(sorted(ready))
        next_ready = []
        for u in ready:
            level_of[u] = k
        for u in ready:
            for v in consumers_of_node[u]:
                indeg_from_nodes[v] -= 1
                if indeg_from_nodes[v]==0:
                    next_ready.append(v)
        ready = next_ready
        k += 1

    for i in range(N):
        if level_of[i] < 0:
            level_of[i] = k
    return levels, indeg_from_nodes, level_of

def assemble_edges(model) -> List[Dict]:
    producer_of, _, _ = build_prod_cons_maps(model)
    edges: Dict[Tuple[int,int], List[str]] = {}
    for dst_idx, node in enumerate(model.graph.node):
        for x in node.input:
            if x and x in producer_of:
                src_idx = producer_of[x]
                edges.setdefault((src_idx, dst_idx), []).append(x)
    out = []
    for (s,d), ts in edges.items():
        out.append({"src_idx": s, "dst_idx": d, "tensors": ts})
    out.sort(key=lambda e: (e["src_idx"], e["dst_idx"]))
    return out

def merge_timeline_steps(timeline_json: Optional[str]) -> Dict[int,int]:
    if not timeline_json: return {}
    jp = Path(timeline_json)
    if not jp.exists(): return {}
    with jp.open() as f:
        data = json.load(f)
    steps = {}
    for item in data.get("timeline", []):
        idx = item.get("idx")
        stp = item.get("step")
        if isinstance(idx, int) and isinstance(stp, int):
            steps[idx] = stp
    return steps

def to_json(model, timeline_json: Optional[str], json_out: str, write_dot: bool):
    graph = model.graph
    N = len(graph.node)
    producer_of, consumers_of_node, outputs_of_node = build_prod_cons_maps(model)
    levels, indeg_after_relax, level_of = compute_levels(model)
    edges = assemble_edges(model)
    step_map = merge_timeline_steps(timeline_json)

    nodes = []
    for i, node in enumerate(graph.node):
        ins = [x for x in node.input if x]
        outs = [x for x in node.output if x]
        producers = []
        for x in ins:
            if x in producer_of:
                producers.append(producer_of[x])
        consumers = consumers_of_node.get(i, [])
        nodes.append({
            "idx": i,
            "name": node.name or f"node_{i}",
            "op_type": node.op_type,
            "inputs": ins,
            "outputs": outs,
            "producers": sorted(set(producers)),
            "consumers": sorted(set(consumers)),
            "fanin_from_nodes": len(set(producers)),
            "fanout": len(set(consumers)),
            "level": level_of[i],
            **({"step": step_map[i]} if i in step_map else {}),
        })

    summary = {
        "model_path": None if not graph.name else graph.name,
        "num_nodes": N,
        "levels": levels,
        "nodes": nodes,
        "edges": edges
    }

    outp = Path(json_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        json.dump(summary, f, indent=2)

    if write_dot:
        dot_lines = ["digraph onnx_deps {", "  rankdir=LR;"]
        for n in nodes:
            dot_lines.append(f'  n{n["idx"]} [label="{n["idx"]}\\n{n["op_type"]}"];')
        for e in edges:
            s, d = e["src_idx"], e["dst_idx"]
            dot_lines.append(f"  n{s} -> n{d};")
        dot_lines.append("}")
        dot_path = outp.with_suffix(".dot")
        with dot_path.open("w") as f:
            f.write("\n".join(dot_lines))
        print(f"[DOT] {dot_path}")

    print(f"[JSON] 완료 -> {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to ONNX model")
    ap.add_argument("--timeline", type=str, default=None, help="(optional) timeline JSON to attach steps")
    ap.add_argument("--json-out", type=str, default=None, help="Output JSON (default: deps/<stem>_deps.json)")
    ap.add_argument("--dot", action="store_true", help="Also write GraphViz DOT next to JSON")
    args = ap.parse_args()

    mp = Path(args.model)
    model = load_onnx(args.model)

    json_out = args.json_out or str(Path("deps") / f"{_stem(mp)}_deps.json")
    to_json(model, args.timeline, json_out, args.dot)

if __name__ == "__main__":
    main()

