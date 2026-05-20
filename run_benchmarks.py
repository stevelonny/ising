#!/usr/bin/env python3
"""
Condensed Ising GPU Benchmark Suite
Compares global-memory vs shared-memory CUDA implementations
"""

import json
import os
from io import StringIO
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    missing = str(e)
    sys.stderr.write(f"Missing Python dependencies for run_benchmarks.py: {missing}\n")
    sys.stderr.write("Install them with:\n  pip install -r requirements.txt\nOr run setup_venv.sh to create a venv with the required packages.\n")
    raise


def run_cmd(cmd, cwd=None, timeout=600):
    """Execute shell command and return (returncode, stdout, stderr)"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def setup_environment():
    """Verify CUDA toolchain and extract compute capability"""
    print('Verifying CUDA toolchain...')
    rc, out, err = run_cmd('nvcc --version', timeout=10)
    if rc != 0:
        raise RuntimeError(f'nvcc not available: {err}')
    print(out)
    
    rc, out, err = run_cmd('nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader', timeout=10)
    if rc != 0:
        raise RuntimeError(f'nvidia-smi not available: {err}')
    print(out)
    compute_capability = out.strip().split(',')[1].strip().replace('.', '')
    print(f'Using sm_{compute_capability} for compilation')
    return compute_capability


def compile_benchmark(workspace, compute_capability):
    """Compile bench_comparison.cu to bench_comparison.x"""
    benchmark_source = workspace / 'bench_comparison.cu'
    benchmark_binary = workspace / 'bench_comparison.x'
    
    if not benchmark_source.exists():
        raise FileNotFoundError(f'Missing benchmark source: {benchmark_source}')
    
    compile_cmd = f'nvcc -O3 -std=c++17 -arch=sm_{compute_capability} {shlex.quote(str(benchmark_source))} -o {shlex.quote(str(benchmark_binary))} -lcurand'
    print(f'\nCompiling: {compile_cmd}')
    rc, out, err = run_cmd(compile_cmd, cwd=str(workspace), timeout=1200)
    
    if rc != 0:
        print(out)
        print(err)
        raise RuntimeError('Compilation failed')
    print(out)
    print(f'Compiled binary: {benchmark_binary}')
    return benchmark_binary


def parse_csv_stdout(stdout):
    df = pd.read_csv(StringIO(stdout))
    # normalize columns to expected names used downstream
    df = df.rename(columns={
        'implementation': 'impl',
        'magnetization': 'mag',
        'time_ms': 'elapsed_ms',
        'repeat': 'repeat',
        'size': 'size',
        'sweeps': 'sweeps'
    })
    # ensure mag is numeric: accept 'Yes'/'No' or numeric
    def norm_mag(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (int, float)):
            return int(x)
        xs = str(x).strip()
        if xs.lower() in ('yes', '1', 'true'):
            return 1
        if xs.lower() in ('no', '0', 'false'):
            return 0
        try:
            return int(xs)
        except Exception:
            return 0
    df['mag'] = df['mag'].apply(norm_mag)
    # normalize implementation names
    df['impl'] = df['impl'].astype(str).str.replace('GPU_Global', 'global')
    df['impl'] = df['impl'].astype(str).str.replace('Shared_Memory', 'shared')
    df['impl'] = df['impl'].astype(str).str.replace('shared_cooperative', 'shared_cooperative')
    # add boundary condition column (bench_comparison uses Periodic only)
    df['bc'] = 'Periodic'
    return df.to_dict(orient='records')


def run_benchmark(benchmark_binary, impl, bc, size, sweeps, mag, repeats, workspace):
    """Run a single benchmark configuration"""
    cmd = [
        str(benchmark_binary),
        '--impl', impl,
        '--bc', bc,
        '--size', str(size),
        '--sweeps', str(sweeps),
        '--mag', str(int(mag)),
        '--repeats', str(repeats),
    ]
    rc, out, err = run_cmd(' '.join(shlex.quote(part) for part in cmd), cwd=str(workspace), timeout=7200)
    if rc != 0:
        raise RuntimeError(err or out or 'benchmark failed')
    return parse_csv_stdout(out)


def sweeps_for_size(size):
    """Adaptive sweep schedule based on lattice size"""
    if size <= 128:
        return 2048
    if size <= 512:
        return 1024
    if size <= 1024:
        return 512
    if size <= 2048:
        return 256
    if size <= 4096:
        return 128
    return 64


def main():
    """Main benchmark execution"""
    # Make the script directory-agnostic: use the script's parent directory
    workspace = Path(__file__).resolve().parent
    results_dir = workspace / 'bench_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    sizes = [1024, 2048, 4096, 8192]
    max_size_env = os.getenv('BENCH_MAX_SIZE')
    if max_size_env:
        max_size = int(max_size_env)
        sizes = [size for size in sizes if size <= max_size]
    implementations = ['global', 'shared', 'shared_cooperative']
    boundary_conditions = ['Periodic']
    repeats = int(os.getenv('BENCH_REPEATS', '5'))
    
    sweeps_by_size = {size: sweeps_for_size(size) for size in sizes}
    
    print('=' * 80)
    print('ISING GPU BENCHMARK SUITE')
    print('=' * 80)
    print(f'Results directory: {results_dir}')
    print(f'Configurations: {len(implementations)} implementations × {len(sizes)} sizes × {len(boundary_conditions)} BC × repeats={repeats}')

    benchmark_plan = []
    for bc in boundary_conditions:
        for impl in implementations:
            #mags = [0, 1] if impl != 'shared_cooperative' else [1]
            mags = [1]
            for mag in mags:
                for size in sizes:
                    benchmark_plan.append((impl, bc, size, sweeps_by_size[size], mag))
    print(f'Total benchmark runs: {len(benchmark_plan)}')
    
    # Setup
    compute_capability = setup_environment()
    benchmark_binary = compile_benchmark(workspace, compute_capability)

    # Run per-configuration benchmarks
    print('\n' + '=' * 80)
    print('RUNNING BENCHMARKS')
    print('=' * 80)

    results = []
    for impl, bc, size, sweeps, mag in benchmark_plan:
        print(f'  {impl:18s} {bc:8s} size={size:5d} sweeps={sweeps:4d} mag={mag}...', end=' ', flush=True)
        rows = run_benchmark(benchmark_binary, impl, bc, size, sweeps, mag, repeats, workspace)
        results.extend(rows)
        print(f'✓ ({len(rows)} samples)')
    
    # Aggregate
    print('\n' + '=' * 80)
    print('AGGREGATING RESULTS')
    print('=' * 80)
    
    df = pd.DataFrame(results)
    df['elapsed_s'] = df['elapsed_ms'] / 1000.0
    df['throughput'] = (df['size'] * df['size'] * df['sweeps']) / df['elapsed_s']
    
    agg = df.groupby(['impl', 'bc', 'size', 'mag']).agg({
        'elapsed_ms': ['mean', 'std', 'median'],
        'throughput': ['mean', 'std'],
        'sweeps': 'first'
    }).reset_index()
    agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg.columns]
    
    csv_path = results_dir / 'results.csv'
    agg_csv_path = results_dir / 'results_aggregated.csv'
    json_path = results_dir / 'results.json'
    
    df.to_csv(csv_path, index=False)
    agg.to_csv(agg_csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))
    
    print(f'Raw results: {len(df)} rows → {csv_path}')
    print(f'Aggregated: {len(agg)} rows → {agg_csv_path}')
    print(f'JSON: {json_path}')
    
    # Plot
    print('\n' + '=' * 80)
    print('GENERATING ANALYSIS PLOTS')
    print('=' * 80)
    
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Elapsed time
    impls_plot = [('global', 'o'), ('shared', 's'), ('shared_cooperative', '^')]
    for mag, linestyle in [(0, '-'), (1, '--')]:
        for impl, marker in impls_plot:
            subset = agg[(agg['mag'] == mag) & (agg['impl'] == impl)].sort_values('size')
            label = f'{impl} mag={mag}'
            axes[0, 0].errorbar(subset['size'], subset['elapsed_ms_mean'], 
                               yerr=subset['elapsed_ms_std'], marker=marker, 
                               linestyle=linestyle, capsize=3, label=label)
    
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Elapsed Time vs Lattice Size')
    axes[0, 0].set_xlabel('Size')
    axes[0, 0].set_ylabel('Elapsed (ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Throughput
    for mag, linestyle in [(0, '-'), (1, '--')]:
        for impl, marker in impls_plot:
            subset = agg[(agg['mag'] == mag) & (agg['impl'] == impl)].sort_values('size')
            label = f'{impl} mag={mag}'
            axes[0, 1].errorbar(subset['size'], subset['throughput_mean'], 
                               yerr=subset['throughput_std'], marker=marker, 
                               linestyle=linestyle, capsize=3, label=label)
    
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Throughput vs Lattice Size')
    axes[0, 1].set_xlabel('Size')
    axes[0, 1].set_ylabel('Spin Updates / s')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Speedup (global vs each shared variant)
    speedup_rows = []
    shared_variants = ['shared', 'shared_cooperative']
    for bc in boundary_conditions:
        for mag in [0, 1]:
            g = agg[(agg['impl'] == 'global') & (agg['bc'] == bc) & (agg['mag'] == mag)][['size', 'elapsed_ms_mean']].rename(columns={'elapsed_ms_mean': 'global_ms'})
            for s_impl in shared_variants:
                s = agg[(agg['impl'] == s_impl) & (agg['bc'] == bc) & (agg['mag'] == mag)][['size', 'elapsed_ms_mean']].rename(columns={'elapsed_ms_mean': 'shared_ms'})
                if s.empty or g.empty:
                    continue
                merged = g.merge(s, on='size')
                merged['speedup'] = merged['global_ms'] / merged['shared_ms']
                merged['mag'] = mag
                merged['variant'] = s_impl
                speedup_rows.append(merged)

    if speedup_rows:
        speedup_df = pd.concat(speedup_rows, ignore_index=True)
        for variant, marker in [('shared', 'o'), ('shared_cooperative', 's')]:
            subset = speedup_df[speedup_df['variant'] == variant].sort_values('size')
            if subset.empty:
                continue
            axes[1, 0].semilogx(subset['size'], subset['speedup'], marker=marker, label=f'{variant}')
    
    axes[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=1, label='baseline')
    axes[1, 0].set_title('Global / Shared Speedup')
    axes[1, 0].set_xlabel('Size')
    axes[1, 0].set_ylabel('Time Global / Time Shared')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = 'Summary Statistics\n' + '=' * 40 + '\n'
    summary_text += 'Mean elapsed time (ms) by impl & mag:\n'
    for (impl, mag), val in agg.groupby(['impl', 'mag'])['elapsed_ms_mean'].mean().items():
        summary_text += f'  {impl:8s} mag={mag}: {val:10.2f}\n'
    summary_text += '\nMean throughput (Mspins/s) by impl & mag:\n'
    for (impl, mag), val in (agg.groupby(['impl', 'mag'])['throughput_mean'].mean() / 1e6).items():
        summary_text += f'  {impl:8s} mag={mag}: {val:10.2f}\n'
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    fig_path = results_dir / 'comparison.png'
    plt.savefig(fig_path, dpi=160, bbox_inches='tight')
    print(f'Plot saved: {fig_path}')
    plt.show()
    
    # Final summary
    print('\n' + '=' * 80)
    print('BENCHMARK COMPLETE')
    print('=' * 80)
    print(f'Results saved to: {results_dir}')
    print(f'  - {csv_path}')
    print(f'  - {agg_csv_path}')
    print(f'  - {json_path}')
    print(f'  - {fig_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
