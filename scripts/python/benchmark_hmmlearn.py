#!/usr/bin/env python3
"""
benchmark_hmmlearn.py
Benchmark hmmlearn (GaussianHMM) sur les mêmes données que CPU/GPU pour comparaison.

Usage:
    python benchmark_hmmlearn.py <data_dir> <output_prefix>
    python benchmark_hmmlearn.py data/bench results/hmmlearn_benchmark

Génère:
    - <output_prefix>_results.csv
    - <output_prefix>_results.json
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import sys
from hmmlearn import hmm
# Suppress convergence warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)



@dataclass
class HMMLearnResult:
    dataset_name: str
    algo_name: str
    T: int
    N: int
    K: int
    time_ms: float
    time_std_ms: float
    log_likelihood: float
    iterations: int
    converged: bool


def load_benchmark_data(base_path: str) -> dict:
    """Load benchmark data from binary files."""
    base = Path(base_path)
    
    # Load dimensions
    with open(f"{base}_dims.txt", 'r') as f:
        T, N, K = map(int, f.read().strip().split())
    
    # Load binary files
    obs = np.fromfile(f"{base}_obs.bin", dtype=np.float32).reshape(T, K)
    pi = np.fromfile(f"{base}_pi.bin", dtype=np.float32)
    A = np.fromfile(f"{base}_A.bin", dtype=np.float32).reshape(N, N)
    mu = np.fromfile(f"{base}_mu.bin", dtype=np.float32).reshape(N, K)
    Sigma = np.fromfile(f"{base}_sigma.bin", dtype=np.float32).reshape(N, K, K)
    
    return {
        'name': base.name,
        'T': T, 'N': N, 'K': K,
        'obs': obs,
        'pi': pi,
        'A': A,
        'mu': mu,
        'Sigma': Sigma
    }


def profile_hmmlearn_fit(data: dict, n_iter: int = 10, num_runs: int = 3) -> HMMLearnResult:
    """Profile GaussianHMM.fit() with fixed iterations."""
    
    result = HMMLearnResult(
        dataset_name=data['name'],
        algo_name=f"hmmlearn_fit_{n_iter}",
        T=data['T'],
        N=data['N'],
        K=data['K'],
        time_ms=0,
        time_std_ms=0,
        log_likelihood=0,
        iterations=n_iter,
        converged=False
    )
    
    times = []
    final_ll = 0
    
    for run in range(num_runs):
        # Create fresh model with initial parameters
        model = hmm.GaussianHMM(
            n_components=data['N'],
            covariance_type='full',
            n_iter=n_iter,
            tol=1e-10,  # Very small to force all iterations
            init_params='',  # Don't re-init, use our params
            params='stmc'  # Update all params
        )
        
        # Set initial parameters
        model.startprob_ = data['pi'].copy()
        model.transmat_ = data['A'].copy()
        model.means_ = data['mu'].copy()
        model.covars_ = data['Sigma'].copy()
        
        # Time the fit
        start = time.perf_counter()
        model.fit(data['obs'])
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        times.append(elapsed_ms)
        final_ll = model.score(data['obs'])
    
    result.time_ms = np.mean(times)
    result.time_std_ms = np.std(times) if len(times) > 1 else 0
    result.log_likelihood = final_ll
    
    return result


def profile_hmmlearn_convergence(data: dict, tolerance: float = 1e-4, 
                                  max_iter: int = 100) -> HMMLearnResult:
    """Profile GaussianHMM.fit() until convergence."""
    
    result = HMMLearnResult(
        dataset_name=data['name'],
        algo_name="hmmlearn_converge",
        T=data['T'],
        N=data['N'],
        K=data['K'],
        time_ms=0,
        time_std_ms=0,
        log_likelihood=0,
        iterations=0,
        converged=False
    )
    
    # Create model
    model = hmm.GaussianHMM(
        n_components=data['N'],
        covariance_type='full',
        n_iter=max_iter,
        tol=tolerance,
        init_params='',
        params='stmc'
    )
    
    # Set initial parameters
    model.startprob_ = data['pi'].copy()
    model.transmat_ = data['A'].copy()
    model.means_ = data['mu'].copy()
    model.covars_ = data['Sigma'].copy()
    
    # Time the fit
    start = time.perf_counter()
    model.fit(data['obs'])
    result.time_ms = (time.perf_counter() - start) * 1000
    
    result.log_likelihood = model.score(data['obs'])
    result.iterations = model.monitor_.iter
    result.converged = model.monitor_.converged
    
    return result


def profile_hmmlearn_viterbi(data: dict, num_runs: int = 10) -> HMMLearnResult:
    """Profile GaussianHMM.decode() (Viterbi)."""
    
    result = HMMLearnResult(
        dataset_name=data['name'],
        algo_name="hmmlearn_viterbi",
        T=data['T'],
        N=data['N'],
        K=data['K'],
        time_ms=0,
        time_std_ms=0,
        log_likelihood=0,
        iterations=1,
        converged=True
    )
    
    # Create and fit model first (or use initial params)
    model = hmm.GaussianHMM(
        n_components=data['N'],
        covariance_type='full',
        n_iter=1,
        init_params=''
    )
    
    model.startprob_ = data['pi'].copy()
    model.transmat_ = data['A'].copy()
    model.means_ = data['mu'].copy()
    model.covars_ = data['Sigma'].copy()
    
    # Warmup
    model.decode(data['obs'])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ll, path = model.decode(data['obs'])
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        result.log_likelihood = ll
    
    result.time_ms = np.mean(times)
    result.time_std_ms = np.std(times) if len(times) > 1 else 0
    
    return result


def profile_hmmlearn_score(data: dict, num_runs: int = 10) -> HMMLearnResult:
    """Profile GaussianHMM.score() (Forward algorithm for LL)."""
    
    result = HMMLearnResult(
        dataset_name=data['name'],
        algo_name="hmmlearn_score",
        T=data['T'],
        N=data['N'],
        K=data['K'],
        time_ms=0,
        time_std_ms=0,
        log_likelihood=0,
        iterations=1,
        converged=True
    )
    
    model = hmm.GaussianHMM(
        n_components=data['N'],
        covariance_type='full',
        n_iter=1,
        init_params=''
    )
    
    model.startprob_ = data['pi'].copy()
    model.transmat_ = data['A'].copy()
    model.means_ = data['mu'].copy()
    model.covars_ = data['Sigma'].copy()
    
    # Warmup
    model.score(data['obs'])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ll = model.score(data['obs'])
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
        result.log_likelihood = ll
    
    result.time_ms = np.mean(times)
    result.time_std_ms = np.std(times) if len(times) > 1 else 0
    
    return result


def export_results(results: List[HMMLearnResult], output_prefix: str):
    """Export results to CSV and JSON."""
    
    # CSV
    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = f"{output_prefix}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # JSON
    json_path = f"{output_prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'hmmlearn_profiling_results': [asdict(r) for r in results]
        }, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark hmmlearn on HMM datasets')
    parser.add_argument('data_dir', help='Directory containing benchmark data')
    parser.add_argument('output_prefix', help='Prefix for output files')
    parser.add_argument('--mode', choices=['all', 'scaling', 'convergence', 'quick'],
                        default='all', help='Test mode (default: all)')
    
    args = parser.parse_args()
    
    
    
    data_dir = Path(args.data_dir)
    output_prefix = args.output_prefix
    mode = args.mode
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       HMMLEARN BENCHMARK - GaussianHMM Profiling             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Data dir: {str(data_dir):<47}║")
    print(f"║  Output:   {output_prefix:<47}║")
    print(f"║  Mode:     {mode:<47}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Find datasets
    dataset_bases = []
    for f in data_dir.glob("*_obs.bin"):
        base = str(f).replace("_obs.bin", "")
        dataset_bases.append(base)
    
    dataset_bases.sort()
    
    print(f"Found {len(dataset_bases)} datasets:")
    for base in dataset_bases:
        print(f"  - {Path(base).name}")
    print()
    
    results = []
    
    for base_path in dataset_bases:
        try:
            data = load_benchmark_data(base_path)
        except Exception as e:
            print(f"Failed to load {base_path}: {e}")
            continue
        
        # Skip large datasets in quick mode
        if mode == 'quick' and data['T'] > 10000:
            continue
        
        print("━" * 64)
        print(f"Profiling: {data['name']} (T={data['T']}, N={data['N']}, K={data['K']})")
        print("━" * 64)
        
        # Adjust runs based on size
        num_runs = 3 if data['T'] > 50000 else 10
        bw_runs = 1 if data['T'] > 20000 else 3
        
        if mode in ['all', 'scaling', 'quick']:
            # Score (Forward)
            print("  [Score/Forward] ", end='', flush=True)
            r_score = profile_hmmlearn_score(data, num_runs)
            print(f"{r_score.time_ms:.2f} ms (±{r_score.time_std_ms:.2f})")
            results.append(r_score)
            
            # Viterbi
            print("  [Viterbi/Decode] ", end='', flush=True)
            r_vit = profile_hmmlearn_viterbi(data, num_runs)
            print(f"{r_vit.time_ms:.2f} ms (±{r_vit.time_std_ms:.2f})")
            results.append(r_vit)
            
            # Baum-Welch 10 iterations
            print("  [Fit 10 iter] ", end='', flush=True)
            r_bw10 = profile_hmmlearn_fit(data, n_iter=10, num_runs=bw_runs)
            print(f"{r_bw10.time_ms:.2f} ms (±{r_bw10.time_std_ms:.2f})")
            results.append(r_bw10)
        
        if mode in ['all', 'convergence']:
            # Convergence test on small datasets
            if 'validation' in data['name'] or 'scaling_T_1000' in data['name']:
                print("  [Fit Convergence] ", end='', flush=True)
                r_conv = profile_hmmlearn_convergence(data, tolerance=1e-4, max_iter=100)
                conv_str = "✓" if r_conv.converged else "✗"
                print(f"{r_conv.time_ms:.2f} ms ({r_conv.iterations} iters, {conv_str})")
                results.append(r_conv)
        
        print()
    
    # Export
    print("═" * 64)
    print("                       EXPORT RESULTS                          ")
    print("═" * 64)
    
    # Create output directory if needed
    output_path = Path(output_prefix)
    if output_path.parent.name:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_results(results, output_prefix)
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    BENCHMARK COMPLETE                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Total tests: {len(results):<45}║")
    print(f"║  CSV output:  {output_prefix + '_results.csv':<45}║")
    print(f"║  JSON output: {output_prefix + '_results.json':<45}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("Next steps:")
    print("  Analyze with Python:")
    print(f"    python analyze_benchmark_results.py --hmmlearn-csv {output_prefix}_results.csv")
    print()


if __name__ == '__main__':
    main()