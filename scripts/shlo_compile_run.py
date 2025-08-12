#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_xla.py – Run benchmarks on **PyTorch-XLA StableHLO bundles**  
(TPU / XLA-CPU / PJRT-GPU). 10-step warm-up + 10-step measurement,  
avg/min/max latency, (가능 시) peak GPU 메모리 측정, CSV 저장.

폴더 구조
────────
results/xla/StableHLO/{model_name}_stablehlo/
  ├─ functions/forward.bytecode …
  └─ …

변경 사항
────────
1. model 이름에서 뒤의 *_block* 을 제거해 ‘conv’, ‘resnet’ 식으로 사용.  
2. CSV에 **input_path** 열 추가(StableHLO 절대경로).  
3. `--dump` 플래그(기본 False) 추가. true 일 때만 XLA dump-옵션 활성.
"""

from __future__ import annotations

import argparse, csv, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Union

# ──────────────────────── 0) 조기 인자파싱: --dump 여부만 판단 ────────────
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument("--dump", action="store_true")
EARLY_ARGS, _ = early_parser.parse_known_args()

# ───────────────────────────────────────────── repo & 결과 디렉터리 설정 ──
ROOT_DIR   = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "xla"
STABLE_DIR  = RESULTS_DIR / "StableHLO"
DUMP_DIR    = RESULTS_DIR / "dump_shlo/proto"
for d in (RESULTS_DIR, STABLE_DIR, DUMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.extend([str(ROOT_DIR), str(ROOT_DIR / "scripts")])
from scripts.pytorch_baseline import load_model, _discover_models  # noqa: E402

# ───────────────────────────────────────────── PJRT & XLA 환경변수 설정 ──
os.environ.setdefault("PJRT_DEVICE", "CUDA")
os.environ.setdefault("XLA_PJRT_GPU_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PJRT_GPU_PINNED_MEMORY_POOL_SIZE", "256M")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

if EARLY_ARGS.dump:  # ← dump 옵션이 있을 때만 플래그 구성
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={DUMP_DIR} "
        f"--xla_dump_hlo_pass_re=.* "
        "--xla_dump_hlo_as_proto "
        # f"--xla_dump_hlo_as_text "
        # f"--xla_dump_hlo_as_dot=true"
    )

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.stablehlo import StableHLOGraphModule

import torch_xla.debug.metrics as met

# ───────────────────────────────────────────── Helper Functions ───────────
def _memory_used_mb(device) -> Union[float, None]:
    try:
        info = xm.get_memory_info(device)
    except Exception as e:
        print(f"[MemInfo Error] {type(e).__name__}: {e}", file=sys.stderr)
        return None
    if isinstance(info, dict) and "bytes_used" in info:            # PJRT-GPU
        return info["bytes_used"] / 1024**2
    if isinstance(info, dict) and "kb_total" in info:              # TPU(XRT)
        return (info["kb_total"] - info["kb_free"]) / 1024.0
    if isinstance(info, (tuple, list)) and len(info) == 2:         # TPU SPMD
        kb_free, kb_total = info
        return (kb_total - kb_free) / 1024.0
    print(f"[MemInfo Error] Unexpected format: {info}", file=sys.stderr)
    return None


def _fmt_line(name, device, avg, mn, mx, mem, pad):
    mem_str = f"{mem:8.2f} MB" if mem is not None else "   n/a"
    return (f"{name:<{pad}s} | {device:<7s} | "
            f"{avg:>10.3f} / {mn:>10.3f} / {mx:>10.3f} ms | {mem_str}")


def _make_inputs(dummy: Any, device):
    if isinstance(dummy, tuple):
        if all(isinstance(t, torch.Tensor) for t in dummy):
            return tuple(t.to(device) for t in dummy)
        return (torch.randn(*dummy, device=device),)
    if isinstance(dummy, dict):
        return {k: v.to(device) for k, v in dummy.items()}
    return (dummy.to(device),)


def _measure(runner,
             dummy,
             *,
             device,
             warmup: int,
             repeats: int) -> Tuple[float, float, float, Union[float, None]]:

    if hasattr(runner, "to"):   runner = runner.to(device)
    if hasattr(runner, "eval"): runner.eval()

    inputs = _make_inputs(dummy, device)


    # Call whole graph module
    call_fn = lambda: runner(*inputs) 


    # def _call():
    #     with torch.no_grad():
    #         if isinstance(inputs, dict):
    #             return runner(**inputs)
    #         if isinstance(inputs, tuple):
    #             return runner(*inputs)
    #         return runner(inputs)

    # warm-up = compile
    for _ in range(warmup):
        call_fn(); xm.mark_step()
    xm.wait_device_ops()

    # Initiate metrics and IR
    met.clear_counters()
    torch_xla._XLAC._clear_pending_irs(str(device))


    # xm.mark_step() ; xm.wait_device_ops()  # Warm-up

    # Measure
    t0 = time.perf_counter()

    for _ in range(repeats):
        call_fn(); xm.mark_step()

    xm.wait_device_ops()
    total_ms = (time.perf_counter() - t0) * 1000.0

    avg_ms = total_ms / repeats
    min_ms = max_ms = avg_ms  # 초기값

    peak_mb = _memory_used_mb(device)

    # for _ in range(repeats):
    #     t0 = time.perf_counter()
    #     out = _call(); xm.mark_step()
    #     xm.wait_device_ops()
    #     times.append((time.perf_counter() - t0) * 1000.0)
    #     cur = _memory_used_mb(device)
    #     if peak is not None and cur is not None:
    #         peak = max(peak, cur)

    # D2H copy (최적화 방지)
    # if isinstance(out, torch.Tensor):
    #     out.cpu()
    # elif isinstance(out, (tuple, list)):
    #     [o.cpu() for o in out if isinstance(o, torch.Tensor)]

    return avg_ms, min_ms, max_ms, peak_mb

# ──────────────────────────────────────────────── main ────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="*", help="model keys; omit for ALL")
    parser.add_argument("--device", default="xla:1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--csv_path")
    parser.add_argument("--dump", action="store_true",
                        help="Enable XLA IR dump(StableHLO/HLO/Graphviz)")
    args = parser.parse_args()

    # (dump 플래그는 초기 파싱에서 이미 반영)

    device = xm.xla_device() if args.device.startswith("xla") else torch.device(args.device)

    orig_models = args.model or _discover_models()
    models = [m[:-6] if m.endswith("_block") else m for m in orig_models]  # 1️⃣ suffix 제거
    pad = max(len(m) for m in models) + 2

    hdr = ("model".ljust(pad) +
           " | device  |    avg /     min /     max (ms) | peak MB")
    print("─" * len(hdr)); print(hdr); print("─" * len(hdr))

    ts = datetime.now().isoformat(timespec="seconds")
    csv_header = ["timestamp", "model_name", "device", "input_path",  # 2️⃣ 새 열
                  "warmup", "repeats",
                  "latency_avg_ms", "latency_min_ms", "latency_max_ms",
                  "peak_memory_mb"]
    rows: List[List[str]] = []

    for name in models:
        shlo_dir = STABLE_DIR / f"{name}_stablehlo"
        if not shlo_dir.is_dir():
            print(f"[SKIP] {name}: StableHLO dir not found")
            continue

        try:
            _, dummy = load_model(name)  # load_model 자동으로 _block 처리
            runner  = StableHLOGraphModule.load(str(shlo_dir))
            avg, mn, mx, mem = _measure(
                runner, dummy, device=device,
                warmup=args.warmup, repeats=args.repeats)

            print(_fmt_line(name, str(device), avg, mn, mx, mem, pad))
            rows.append([ts, name, str(device), str(shlo_dir.resolve()),
                         args.warmup, args.repeats,
                         f"{avg:.3f}", f"{mn:.3f}", f"{mx:.3f}",
                         f"{mem:.2f}" if mem is not None else "n/a"])
        except Exception as exc:
            reason = str(exc).splitlines()[0]
            print(f"[ERROR] {name}: {reason}")
            rows.append([ts, name, str(device), str(shlo_dir.resolve()),
                         args.warmup, args.repeats,
                         "ERROR", "ERROR", "ERROR", reason])

    # CSV 저장
    tag        = "all" if not args.model else "_".join(models)
    defaultcsv = RESULTS_DIR / f"xla_metrics_{tag}_{device}.csv"
    path       = Path(args.csv_path) if args.csv_path else defaultcsv
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(csv_header)
        csv.writer(f).writerows(rows)

    print("─" * len(hdr))
    print(f"[✓] CSV saved → {path}")

if __name__ == "__main__":
    main()

