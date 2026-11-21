#!/usr/bin/env python3
"""
Ising GPU Scaling Benchmark & Profiler (Shared vs Cooperative)
=============================================================
Automates testing multiple lattice sizes (L = 6, 7, 8, 9, 10, ...) with:
  1. High-throughput performance benchmarks (sweeps/sec, SUPS, speedup)
  2. Targeted Nsight Systems (nsys) timeline & launch overhead profiling
  3. Microarchitectural Nsight Compute (ncu) analysis (roofline & bank conflicts)
  4. Cross-GPU comparison (GTX 1650 vs RTX 2060)

Usage:
  python3 profile_scaling.py --mode bench --sweeps 100000 --sizes 6 7 8 9
  python3 profile_scaling.py --mode nsys --nsys-sweeps 50000 --sizes 6 8
  python3 profile_scaling.py --mode ncu --sizes 6 8
  python3 profile_scaling.py --compare results/benchmark_gtx1650.csv results/benchmark_rtx2060.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def run_command(cmd, cwd=None, capture=True, check=True):
    """Run a shell command and return stdout/stderr."""
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
    
    res = subprocess.run(
        cmd_str,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    if check and res.returncode != 0:
        err_msg = res.stderr if capture else f"exited with code {res.returncode}"
        raise RuntimeError(f"Command failed [{res.returncode}]: {cmd_str}\nError: {err_msg}")
    return res


def get_gpu_info():
    """Detect GPU model, compute capability, SM count, and VRAM."""
    info = {
        "name": "Unknown_GPU",
        "tag": "unknown",
        "sm": "75",
        "sm_count": 0,
        "vram_mb": 0,
        "driver": "Unknown"
    }
    try:
        res = run_command("nvidia-smi --query-gpu=gpu_name,compute_cap,memory.total,driver_version --format=csv,noheader")
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            info["name"] = parts[0]
            info["sm"] = parts[1].replace(".", "")
            info["vram_mb"] = int(re.sub(r"[^\d]", "", parts[2]))
            info["driver"] = parts[3]
            
            # Create tag
            name_lower = parts[0].lower()
            if "1650" in name_lower:
                info["tag"] = "gtx1650"
            elif "2060" in name_lower:
                info["tag"] = "rtx2060"
            else:
                info["tag"] = re.sub(r"[^a-z0-9]", "", name_lower)[:10]
    except Exception as e:
        print(f"[WARN] Failed to query nvidia-smi: {e}")

    # Estimate SM count
    if "1650" in info["tag"]:
        info["sm_count"] = 14
    elif "2060" in info["tag"]:
        info["sm_count"] = 30
    
    return info


def find_tool(tool_name):
    """Find location of nsys or ncu."""
    path = shutil.which(tool_name)
    if path:
        return path
    
    candidates = [
        f"/usr/local/bin/{tool_name}",
        f"/usr/local/cuda/bin/{tool_name}",
        f"/usr/local/cuda-13.2/bin/{tool_name}",
        f"/usr/local/cuda-13.1/bin/{tool_name}",
        f"/usr/local/cuda-12.0/bin/{tool_name}",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def ensure_binaries(workspace, sm_arch, rebuild=False):
    """Build run_shared.x and run_cooperative.x if missing."""
    shared_bin = workspace / "run_shared.x"
    coop_bin = workspace / "run_cooperative.x"
    
    needs_build = rebuild or not shared_bin.exists() or not coop_bin.exists()
    if needs_build:
        print(f"\n[*] Building binaries for sm_{sm_arch}...")
        res = run_command(f"make run_shared.x run_cooperative.x", cwd=str(workspace))
        if res.stdout:
            print(res.stdout.strip())
    
    if not shared_bin.exists() or not coop_bin.exists():
        raise FileNotFoundError(f"Failed to find or build {shared_bin} and {coop_bin}")
    return shared_bin, coop_bin


def get_sweeps_for_size(log2_size, base_sweeps, adaptive=False):
    """Calculate sweeps, optionally scaling down for massive lattices."""
    if not adaptive:
        return base_sweeps
    # Adaptive scaling: keep run times reasonable for L >= 9
    if log2_size <= 7:
        return base_sweeps
    elif log2_size == 8:
        return max(50000, base_sweeps // 2)
    elif log2_size == 9:
        return max(20000, base_sweeps // 5)
    else:
        return max(10000, base_sweeps // 10)


def run_single_benchmark(binary, log2_size, sweeps, temp=2.25, bc="pbc", seed=42, tag="bench"):
    """Run an executable and measure wall-clock duration."""
    cmd = [
        str(binary),
        "-L", str(log2_size),
        "-T", str(temp),
        "-N", str(sweeps),
        "-b", bc,
        "-t", tag,
        "-s", str(seed),
        "-v"
    ]
    t0 = time.perf_counter()
    res = run_command(cmd, check=True)
    t1 = time.perf_counter()
    duration = t1 - t0
    
    # Check for thermalization output
    therm_sweeps = 0
    for line in res.stdout.split("\n"):
        m = re.search(r"Thermalization\s+(\w+)\s+with\s+(\d+)\s+sweeps", line)
        if m:
            therm_sweeps = int(m.group(2))
            
    return duration, therm_sweeps, res.stdout


def mode_benchmark(workspace, gpu_info, sizes, sweeps, adaptive, temp, bc, seed):
    """Run throughput benchmark across multiple lattice sizes."""
    print(f"\n=================================================================")
    print(f"  BENCHMARK MODE: {gpu_info['name']} ({gpu_info['tag'].upper()})")
    print(f"  SMs: {gpu_info['sm_count']} | VRAM: {gpu_info['vram_mb']} MB | Driver: {gpu_info['driver']}")
    print(f"  Lattice sizes L (log2): {sizes}")
    print(f"  Base sweeps: {sweeps} (Adaptive: {adaptive})")
    print(f"=================================================================\n")

    results_dir = workspace / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    csv_file = results_dir / f"benchmark_{gpu_info['tag']}.csv"

    rows = []
    
    header = [
        "gpu", "log2_L", "grid_size", "num_spins", "sweeps",
        "shared_time_s", "coop_time_s", "speedup",
        "shared_sps", "coop_sps", "shared_sups", "coop_sups"
    ]

    print(f"{'L':<4} {'Grid':<10} {'Sweeps':<9} {'Shared (s)':<12} {'Coop (s)':<12} {'Speedup':<9} {'Coop SUPS':<12}")
    print("-" * 74)

    for L in sizes:
        grid_dim = 2 ** L
        num_spins = grid_dim * grid_dim
        actual_sweeps = get_sweeps_for_size(L, sweeps, adaptive)
        
        # 1. Run Shared
        t_shared, therm_s, _ = run_single_benchmark(
            workspace / "run_shared.x", L, actual_sweeps, temp, bc, seed, tag=f"bench_shared_{gpu_info['tag']}"
        )
        
        # 2. Run Cooperative
        t_coop, therm_c, _ = run_single_benchmark(
            workspace / "run_cooperative.x", L, actual_sweeps, temp, bc, seed, tag=f"bench_coop_{gpu_info['tag']}"
        )
        
        speedup = t_shared / t_coop if t_coop > 0 else 0.0
        shared_sps = actual_sweeps / t_shared
        coop_sps = actual_sweeps / t_coop
        shared_sups = (actual_sweeps * num_spins) / t_shared
        coop_sups = (actual_sweeps * num_spins) / t_coop

        print(f"{L:<4} {f'{grid_dim}x{grid_dim}':<10} {actual_sweeps:<9} {t_shared:<12.3f} {t_coop:<12.3f} {speedup:<9.2f}x {coop_sups:<12.2e}")

        rows.append({
            "gpu": gpu_info["tag"],
            "log2_L": L,
            "grid_size": f"{grid_dim}x{grid_dim}",
            "num_spins": num_spins,
            "sweeps": actual_sweeps,
            "shared_time_s": round(t_shared, 4),
            "coop_time_s": round(t_coop, 4),
            "speedup": round(speedup, 2),
            "shared_sps": round(shared_sps, 1),
            "coop_sps": round(coop_sps, 1),
            "shared_sups": f"{shared_sups:.3e}",
            "coop_sups": f"{coop_sups:.3e}"
        })

    # Write CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 74)
    print(f"\n[+] Benchmark complete! Results saved to: {csv_file}\n")


def mode_nsys(workspace, gpu_info, sizes, sweeps, temp, bc, seed):
    """Run Nsight Systems profile on shared vs cooperative."""
    nsys_bin = find_tool("nsys")
    if not nsys_bin:
        print("[ERROR] 'nsys' executable not found in PATH or standard CUDA locations.")
        return

    out_dir = workspace / "results" / gpu_info["tag"] / "nsys"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n=================================================================")
    print(f"  NSIGHT SYSTEMS PROFILING MODE: {gpu_info['name']}")
    print(f"  Target directory: {out_dir}")
    print(f"  Sizes: {sizes} | Sweeps per run: {sweeps}")
    print(f"=================================================================\n")

    for L in sizes:
        grid_dim = 2 ** L
        print(f"\n---> Profiling L={L} ({grid_dim}x{grid_dim}), Sweeps={sweeps}...")

        for impl in ["shared", "cooperative"]:
            bin_name = f"run_{impl}.x"
            rep_base = out_dir / f"nsys_{impl}_L{L}"
            
            # Profile command
            profile_cmd = [
                str(nsys_bin), "profile",
                "--trace=cuda,osrt",
                "--force-overwrite=true",
                "-o", str(rep_base),
                str(workspace / bin_name),
                "-L", str(L),
                "-T", str(temp),
                "-N", str(sweeps),
                "-b", bc,
                "-s", str(seed),
                "-t", "nsys"
            ]
            print(f"  [+] Profiling {bin_name}...")
            run_command(profile_cmd, check=True)

            rep_file = rep_base.with_suffix(".nsys-rep")
            if rep_file.exists():
                print(f"      Report generated: {rep_file.name} ({rep_file.stat().st_size / 1e6:.1f} MB)")
                
                # Extract summary stats
                stats_txt = rep_base.with_name(f"{rep_base.name}_stats.txt")
                stats_cmd = [
                    str(nsys_bin), "stats",
                    "--report", "cuda_gpu_kern_sum,cuda_api_sum",
                    str(rep_file)
                ]
                stats_res = run_command(stats_cmd, check=False)
                with open(stats_txt, "w") as f:
                    f.write(stats_res.stdout)
                print(f"      Summary text saved: {stats_txt.name}")

    print(f"\n[+] Nsys profiling finished. Open `.nsys-rep` files in Nsight Systems GUI.")


def mode_ncu(workspace, gpu_info, sizes, sweeps, temp, bc, seed):
    """Run Nsight Compute microarchitectural profiling."""
    ncu_bin = find_tool("ncu")
    if not ncu_bin:
        print("[ERROR] 'ncu' executable not found in PATH or standard CUDA locations.")
        return

    out_dir = workspace / "results" / gpu_info["tag"] / "ncu"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n=================================================================")
    print(f"  NSIGHT COMPUTE PROFILING MODE: {gpu_info['name']}")
    print(f"  Sizes: {sizes} | Sweeps: {sweeps} (low count to avoid replay hangs)")
    print(f"=================================================================\n")

    for L in sizes:
        grid_dim = 2 ** L
        print(f"\n---> NCU Profiling L={L} ({grid_dim}x{grid_dim})...")
        
        # Profile cooperative
        out_file = out_dir / f"ncu_coop_L{L}"
        ncu_cmd = [
            str(ncu_bin),
            "--set", "detailed",
            "--kernel-name", "regex:cooperativeUpdateTileKernel",
            "--launch-count", "1",
            "-o", str(out_file),
            str(workspace / "run_cooperative.x"),
            "-L", str(L),
            "-T", str(temp),
            "-N", str(sweeps),
            "-b", bc,
            "-s", str(seed)
        ]
        print(f"  [+] Profiling cooperativeUpdateTileKernel with NCU...")
        res = run_command(ncu_cmd, check=False)
        if "ERR_NVGPUCTRPERM" in res.stderr or "ERR_NVGPUCTRPERM" in res.stdout:
            print(f"\n[!] NCU Permission Warning: Non-root profiling restricted (ERR_NVGPUCTRPERM).")
            print(f"    To fix on Linux: run with 'sudo' or set modprobe option:")
            print(f"    sudo sh -c 'echo \"options nvidia NVreg_RestrictProfilingToAdminUsers=0\" > /etc/modprobe.d/nvidia-profiling.conf'")
            print(f"    sudo update-initramfs -u\n")
            return
        elif res.returncode == 0:
            print(f"      Report generated: {out_file.with_suffix('.ncu-rep')}")
        else:
            print(f"      NCU failed: {res.stderr[:200]}")


def mode_compare(csv1_path, csv2_path):
    """Compare two benchmark CSV files (e.g. GTX 1650 vs RTX 2060)."""
    p1 = Path(csv1_path)
    p2 = Path(csv2_path)
    if not p1.exists() or not p2.exists():
        print(f"[ERROR] One or both CSV files do not exist: {p1}, {p2}")
        return

    with open(p1) as f1, open(p2) as f2:
        r1 = list(csv.DictReader(f1))
        r2 = list(csv.DictReader(f2))

    gpu1 = r1[0]["gpu"].upper() if r1 else "GPU 1"
    gpu2 = r2[0]["gpu"].upper() if r2 else "GPU 2"

    map2 = {int(row["log2_L"]): row for row in r2}

    print(f"\n==========================================================================================")
    print(f"  CROSS-GPU COMPARISON: {gpu1} vs {gpu2}")
    print(f"==========================================================================================")
    print(f"{'L':<4} {'Grid':<10} {gpu1 + ' Coop(s)':<16} {gpu2 + ' Coop(s)':<16} {'Coop Speedup':<14} {gpu2 + ' SUPS':<12}")
    print("-" * 88)

    for row1 in r1:
        L = int(row1["log2_L"])
        if L in map2:
            row2 = map2[L]
            t1 = float(row1["coop_time_s"])
            t2 = float(row2["coop_time_s"])
            sp = t1 / t2 if t2 > 0 else 0.0
            print(f"{L:<4} {row1['grid_size']:<10} {t1:<16.3f} {t2:<16.3f} {sp:<14.2f}x {row2['coop_sups']:<12}")

    print("-" * 88)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile Ising model (Shared vs Cooperative) across multiple lattice sizes."
    )
    parser.add_argument("--mode", choices=["bench", "nsys", "ncu", "all"], default="bench",
                        help="Execution mode (default: bench)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 7, 8, 9, 10],
                        help="List of log2 lattice sizes (e.g. 6 7 8 9 10 for sizes 64 to 1024)")
    parser.add_argument("--sweeps", "-N", type=int, default=100000,
                        help="Number of measurement sweeps for benchmark (default: 100000)")
    parser.add_argument("--nsys-sweeps", type=int, default=50000,
                        help="Number of sweeps for nsys profiling (default: 50000)")
    parser.add_argument("--ncu-sweeps", type=int, default=10,
                        help="Number of sweeps for ncu profiling (default: 10)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Scale sweeps down adaptively for larger lattices (L >= 9)")
    parser.add_argument("--temp", "-T", type=float, default=2.25,
                        help="Temperature (default: 2.25)")
    parser.add_argument("--bc", "-b", choices=["pbc", "obc"], default="pbc",
                        help="Boundary conditions (default: pbc)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="PRNG seed (default: 42)")
    parser.add_argument("--gpu-tag", type=str, default=None,
                        help="Explicit GPU tag (e.g. gtx1650, rtx2060). Auto-detected if omitted.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of CUDA binaries before running.")
    parser.add_argument("--compare", nargs=2, metavar=("CSV1", "CSV2"),
                        help="Compare two benchmark CSV files side-by-side.")

    args = parser.parse_args()

    # Compare mode
    if args.compare:
        mode_compare(args.compare[0], args.compare[1])
        return

    workspace = Path(__file__).resolve().parent

    # Detect GPU
    gpu_info = get_gpu_info()
    if args.gpu_tag:
        gpu_info["tag"] = args.gpu_tag

    # Build binaries
    ensure_binaries(workspace, gpu_info["sm"], rebuild=args.rebuild)

    # Dispatch modes
    if args.mode in ("bench", "all"):
        mode_benchmark(workspace, gpu_info, args.sizes, args.sweeps, args.adaptive, args.temp, args.bc, args.seed)
    
    if args.mode in ("nsys", "all"):
        mode_nsys(workspace, gpu_info, args.sizes, args.nsys_sweeps, args.temp, args.bc, args.seed)

    if args.mode in ("ncu", "all"):
        mode_ncu(workspace, gpu_info, args.sizes, args.ncu_sweeps, args.temp, args.bc, args.seed)


if __name__ == "__main__":
    main()
