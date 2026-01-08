#!/usr/bin/env python3
"""
analyze_benchmark_results.py
Analyse et visualisation des résultats de profiling CPU vs GPU vs hmmlearn.

Génère 5 graphes principaux:
1. Temps vs T (log-log) - Scaling avec la longueur de séquence
2. Speedup GPU/CPU vs T
3. Convergence (log-likelihood vs itérations)
4. Métriques GPU (occupancy, bandwidth)
5. Cache efficiency CPU

Usage:
    python analyze_benchmark_results.py --cpu-csv results/cpu_results.csv \
                                        --gpu-csv results/gpu_results.csv \
                                        --hmmlearn-csv results/hmmlearn_results.csv \
                                        --output-dir results/figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = {
    'cpu': '#2ecc71',      # Green
    'gpu': '#e74c3c',      # Red
    'hmmlearn': '#3498db', # Blue
    'forward': '#9b59b6',  # Purple
    'backward': '#f39c12', # Orange
    'viterbi': '#1abc9c',  # Teal
    'baum_welch': '#e67e22' # Dark Orange
}


def load_results(cpu_csv: Optional[str], 
                 gpu_csv: Optional[str], 
                 hmmlearn_csv: Optional[str]) -> Tuple[Optional[pd.DataFrame], 
                                                        Optional[pd.DataFrame], 
                                                        Optional[pd.DataFrame]]:
    """Load all result CSVs."""
    cpu_df = pd.read_csv(cpu_csv) if cpu_csv and Path(cpu_csv).exists() else None
    gpu_df = pd.read_csv(gpu_csv) if gpu_csv and Path(gpu_csv).exists() else None
    hmmlearn_df = pd.read_csv(hmmlearn_csv) if hmmlearn_csv and Path(hmmlearn_csv).exists() else None
    
    if cpu_df is not None:
        print(f"✓ Loaded CPU results: {len(cpu_df)} entries")
    if gpu_df is not None:
        print(f"✓ Loaded GPU results: {len(gpu_df)} entries")
    if hmmlearn_df is not None:
        print(f"✓ Loaded hmmlearn results: {len(hmmlearn_df)} entries")
    
    return cpu_df, gpu_df, hmmlearn_df


def extract_T_from_dataset(dataset_name: str) -> Optional[int]:
    """Extract T value from dataset name like 'scaling_T_10000'."""
    if 'scaling_T_' in dataset_name:
        try:
            return int(dataset_name.split('scaling_T_')[1].split('_')[0])
        except:
            pass
    return None


def filter_scaling_T(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only scaling_T datasets."""
    df = df.copy()
    df['T_extracted'] = df['dataset'].apply(extract_T_from_dataset) if 'dataset' in df.columns else \
                        df['dataset_name'].apply(extract_T_from_dataset)
    return df[df['T_extracted'].notna()].copy()


# =============================================================================
# GRAPH 1: Time vs T (Log-Log)
# =============================================================================

def plot_time_vs_T(cpu_df: Optional[pd.DataFrame],
                   gpu_df: Optional[pd.DataFrame],
                   hmmlearn_df: Optional[pd.DataFrame],
                   algo_filter: str = 'baum_welch_10',
                   output_path: str = 'time_vs_T.png'):
    """
    Plot execution time vs sequence length T (log-log scale).
    Shows scaling behavior for CPU, GPU, and hmmlearn.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Left: Absolute times ===
    ax1 = axes[0]
    
    # CPU
    if cpu_df is not None:
        cpu_scaling = filter_scaling_T(cpu_df)
        cpu_algo = cpu_scaling[cpu_scaling['algo'].str.contains(algo_filter, na=False)]
        if len(cpu_algo) > 0:
            cpu_algo = cpu_algo.sort_values('T_extracted')
            ax1.errorbar(cpu_algo['T_extracted'], cpu_algo['time_ms'],
                        yerr=cpu_algo['time_std_ms'], 
                        fmt='o-', color=COLORS['cpu'], linewidth=2, markersize=8,
                        label='CPU (Hassan)', capsize=3)
    
    # GPU
    if gpu_df is not None:
        gpu_scaling = filter_scaling_T(gpu_df)
        gpu_algo = gpu_scaling[gpu_scaling['algo'].str.contains(algo_filter, na=False)]
        if len(gpu_algo) > 0:
            gpu_algo = gpu_algo.sort_values('T_extracted')
            ax1.errorbar(gpu_algo['T_extracted'], gpu_algo['time_ms'],
                        yerr=gpu_algo['time_std_ms'],
                        fmt='s-', color=COLORS['gpu'], linewidth=2, markersize=8,
                        label='GPU (Hassan)', capsize=3)
    
    # hmmlearn
    if hmmlearn_df is not None:
        hmm_scaling = filter_scaling_T(hmmlearn_df)
        hmm_algo = hmm_scaling[hmm_scaling['algo_name'].str.contains('fit_10', na=False)]
        if len(hmm_algo) > 0:
            hmm_algo = hmm_algo.sort_values('T_extracted')
            ax1.errorbar(hmm_algo['T_extracted'], hmm_algo['time_ms'],
                        yerr=hmm_algo['time_std_ms'],
                        fmt='^-', color=COLORS['hmmlearn'], linewidth=2, markersize=8,
                        label='hmmlearn', capsize=3)
    
    ax1.set_xlabel('Sequence Length T')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Execution Time vs T ({algo_filter})')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add theoretical complexity lines
    if cpu_df is not None and len(cpu_algo) > 0:
        T_vals = cpu_algo['T_extracted'].values
        time_base = cpu_algo['time_ms'].values[0]
        T_base = T_vals[0]
        
        # O(T) line
        y_linear = time_base * (T_vals / T_base)
        ax1.plot(T_vals, y_linear, '--', color='gray', alpha=0.5, label='O(T) ref')
        
        # O(T log T) line
        y_tlogt = time_base * (T_vals / T_base) * (np.log2(T_vals) / np.log2(T_base))
        ax1.plot(T_vals, y_tlogt, ':', color='gray', alpha=0.5, label='O(T log T) ref')
    
    ax1.legend(loc='upper left')
    
    # === Right: Time per iteration ===
    ax2 = axes[1]
    
    if cpu_df is not None and len(cpu_algo) > 0:
        time_per_iter = cpu_algo['time_ms'] / 10  # 10 iterations
        ax2.plot(cpu_algo['T_extracted'], time_per_iter, 
                'o-', color=COLORS['cpu'], linewidth=2, markersize=8, label='CPU')
    
    if gpu_df is not None and len(gpu_algo) > 0:
        time_per_iter = gpu_algo['time_ms'] / 10
        ax2.plot(gpu_algo['T_extracted'], time_per_iter,
                's-', color=COLORS['gpu'], linewidth=2, markersize=8, label='GPU')
    
    if hmmlearn_df is not None and len(hmm_algo) > 0:
        time_per_iter = hmm_algo['time_ms'] / 10
        ax2.plot(hmm_algo['T_extracted'], time_per_iter,
                '^-', color=COLORS['hmmlearn'], linewidth=2, markersize=8, label='hmmlearn')
    
    ax2.set_xlabel('Sequence Length T')
    ax2.set_ylabel('Time per EM Iteration (ms)')
    ax2.set_title('Time per Baum-Welch Iteration')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved: {output_path}")


# =============================================================================
# GRAPH 2: Speedup GPU/CPU vs T
# =============================================================================

def plot_speedup(cpu_df: Optional[pd.DataFrame],
                 gpu_df: Optional[pd.DataFrame],
                 hmmlearn_df: Optional[pd.DataFrame],
                 output_path: str = 'speedup_vs_T.png'):
    """
    Plot speedup of GPU vs CPU and GPU vs hmmlearn.
    """
    if cpu_df is None or gpu_df is None:
        print("⚠ Need both CPU and GPU results for speedup plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    algos_to_compare = [
        ('forward', 'Forward'),
        ('viterbi', 'Viterbi'),
        ('baum_welch_10', 'Baum-Welch (10 iter)')
    ]
    
    # === Left: GPU Speedup over CPU ===
    ax1 = axes[0]
    
    for algo, label in algos_to_compare:
        cpu_scaling = filter_scaling_T(cpu_df)
        gpu_scaling = filter_scaling_T(gpu_df)
        
        cpu_algo = cpu_scaling[cpu_scaling['algo'].str.contains(algo, na=False)]
        gpu_algo = gpu_scaling[gpu_scaling['algo'].str.contains(algo, na=False)]
        
        if len(cpu_algo) > 0 and len(gpu_algo) > 0:
            # Merge on T
            merged = pd.merge(
                cpu_algo[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'cpu_time'}),
                gpu_algo[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'gpu_time'}),
                on='T_extracted'
            )
            merged['speedup'] = merged['cpu_time'] / merged['gpu_time']
            merged = merged.sort_values('T_extracted')
            
            ax1.plot(merged['T_extracted'], merged['speedup'], 
                    'o-', linewidth=2, markersize=8, label=label)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax1.set_xlabel('Sequence Length T')
    ax1.set_ylabel('Speedup (CPU Time / GPU Time)')
    ax1.set_title('GPU Speedup over CPU')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate max speedup
    for algo, label in algos_to_compare:
        cpu_algo = cpu_scaling[cpu_scaling['algo'].str.contains(algo, na=False)]
        gpu_algo = gpu_scaling[gpu_scaling['algo'].str.contains(algo, na=False)]
        if len(cpu_algo) > 0 and len(gpu_algo) > 0:
            merged = pd.merge(
                cpu_algo[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'cpu_time'}),
                gpu_algo[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'gpu_time'}),
                on='T_extracted'
            )
            merged['speedup'] = merged['cpu_time'] / merged['gpu_time']
            max_speedup = merged['speedup'].max()
            max_T = merged.loc[merged['speedup'].idxmax(), 'T_extracted']
            ax1.annotate(f'{max_speedup:.1f}x', 
                        xy=(max_T, max_speedup),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
    
    # === Right: GPU Speedup over hmmlearn ===
    ax2 = axes[1]
    
    if hmmlearn_df is not None:
        hmm_scaling = filter_scaling_T(hmmlearn_df)
        
        comparisons = [
            ('baum_welch_10', 'hmmlearn_fit_10', 'Baum-Welch 10 iter'),
            ('viterbi', 'hmmlearn_viterbi', 'Viterbi'),
        ]
        
        for gpu_algo, hmm_algo, label in comparisons:
            gpu_data = gpu_scaling[gpu_scaling['algo'].str.contains(gpu_algo, na=False)]
            hmm_data = hmm_scaling[hmm_scaling['algo_name'].str.contains(hmm_algo.replace('_', ''), na=False)]
            
            if len(gpu_data) > 0 and len(hmm_data) > 0:
                merged = pd.merge(
                    hmm_data[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'hmm_time'}),
                    gpu_data[['T_extracted', 'time_ms']].rename(columns={'time_ms': 'gpu_time'}),
                    on='T_extracted'
                )
                merged['speedup'] = merged['hmm_time'] / merged['gpu_time']
                merged = merged.sort_values('T_extracted')
                
                ax2.plot(merged['T_extracted'], merged['speedup'],
                        'o-', linewidth=2, markersize=8, label=label)
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline (1x)')
        ax2.set_xlabel('Sequence Length T')
        ax2.set_ylabel('Speedup (hmmlearn Time / GPU Time)')
        ax2.set_title('GPU Speedup over hmmlearn')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No hmmlearn data available',
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved: {output_path}")


# =============================================================================
# GRAPH 3: Convergence (Log-Likelihood vs Iterations)
# =============================================================================

def plot_convergence(cpu_df: Optional[pd.DataFrame],
                     gpu_df: Optional[pd.DataFrame],
                     hmmlearn_df: Optional[pd.DataFrame],
                     output_path: str = 'convergence.png'):
    """
    Compare convergence behavior: final LL and iterations to converge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Left: Final Log-Likelihood comparison ===
    ax1 = axes[0]
    
    # Collect convergence results
    results = []
    
    if cpu_df is not None:
        conv = cpu_df[cpu_df['algo'].str.contains('converge', na=False)]
        for _, row in conv.iterrows():
            results.append({
                'Implementation': 'CPU (Hassan)',
                'Dataset': row.get('dataset', row.get('dataset_name', '')),
                'Log-Likelihood': row['log_likelihood'],
                'Iterations': row['iterations']
            })
    
    if gpu_df is not None:
        conv = gpu_df[gpu_df['algo'].str.contains('converge', na=False)]
        for _, row in conv.iterrows():
            results.append({
                'Implementation': 'GPU (Hassan)',
                'Dataset': row.get('dataset', row.get('dataset_name', '')),
                'Log-Likelihood': row['log_likelihood'],
                'Iterations': row['iterations']
            })
    
    if hmmlearn_df is not None:
        conv = hmmlearn_df[hmmlearn_df['algo_name'].str.contains('converge', na=False)]
        for _, row in conv.iterrows():
            results.append({
                'Implementation': 'hmmlearn',
                'Dataset': row.get('dataset', row.get('dataset_name', '')),
                'Log-Likelihood': row['log_likelihood'],
                'Iterations': row['iterations']
            })
    
    if results:
        df_conv = pd.DataFrame(results)
        
        # Bar plot of final LL
        datasets = df_conv['Dataset'].unique()
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, impl in enumerate(['CPU (Hassan)', 'GPU (Hassan)', 'hmmlearn']):
            impl_data = df_conv[df_conv['Implementation'] == impl]
            if len(impl_data) > 0:
                lls = [impl_data[impl_data['Dataset'] == d]['Log-Likelihood'].values[0] 
                       if len(impl_data[impl_data['Dataset'] == d]) > 0 else 0 
                       for d in datasets]
                color = COLORS['cpu'] if 'CPU' in impl else (COLORS['gpu'] if 'GPU' in impl else COLORS['hmmlearn'])
                ax1.bar(x + i * width, lls, width, label=impl, color=color, alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Final Log-Likelihood')
        ax1.set_title('Convergence: Final Log-Likelihood Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # === Right: Iterations to converge ===
    ax2 = axes[1]
    
    if results:
        for i, impl in enumerate(['CPU (Hassan)', 'GPU (Hassan)', 'hmmlearn']):
            impl_data = df_conv[df_conv['Implementation'] == impl]
            if len(impl_data) > 0:
                iters = [impl_data[impl_data['Dataset'] == d]['Iterations'].values[0] 
                         if len(impl_data[impl_data['Dataset'] == d]) > 0 else 0 
                         for d in datasets]
                color = COLORS['cpu'] if 'CPU' in impl else (COLORS['gpu'] if 'GPU' in impl else COLORS['hmmlearn'])
                ax2.bar(x + i * width, iters, width, label=impl, color=color, alpha=0.8)
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Iterations to Converge')
        ax2.set_title('Convergence: Number of Iterations')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved: {output_path}")


# =============================================================================
# GRAPH 4: GPU Metrics (Bandwidth, Occupancy)
# =============================================================================

def plot_gpu_metrics(gpu_df: Optional[pd.DataFrame],
                     output_path: str = 'gpu_metrics.png'):
    """
    Plot GPU-specific metrics: bandwidth and occupancy.
    """
    if gpu_df is None:
        print("⚠ No GPU data available for metrics plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gpu_scaling = filter_scaling_T(gpu_df)
    
    # === Top Left: Effective Bandwidth vs T ===
    ax1 = axes[0, 0]
    
    for algo in ['forward', 'backward', 'viterbi', 'baum_welch_10']:
        algo_data = gpu_scaling[gpu_scaling['algo'].str.contains(algo, na=False)]
        if len(algo_data) > 0:
            algo_data = algo_data.sort_values('T_extracted')
            ax1.plot(algo_data['T_extracted'], algo_data['bandwidth_GBs'],
                    'o-', linewidth=2, markersize=6, label=algo.replace('_', ' ').title())
    
    ax1.set_xlabel('Sequence Length T')
    ax1.set_ylabel('Effective Bandwidth (GB/s)')
    ax1.set_title('GPU Memory Bandwidth vs T')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Top Right: Bandwidth by Algorithm ===
    ax2 = axes[0, 1]
    
    algos = gpu_df['algo'].unique()
    bw_means = [gpu_df[gpu_df['algo'] == a]['bandwidth_GBs'].mean() for a in algos]
    colors = [COLORS.get(a.split('_')[0], '#95a5a6') for a in algos]
    
    bars = ax2.bar(range(len(algos)), bw_means, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels(algos, rotation=45, ha='right')
    ax2.set_ylabel('Average Bandwidth (GB/s)')
    ax2.set_title('Average Bandwidth by Algorithm')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, bw_means):
        ax2.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # === Bottom Left: GPU Memory Usage ===
    ax3 = axes[1, 0]
    
    if 'gpu_mem_MB' in gpu_scaling.columns:
        for algo in ['forward', 'baum_welch_10']:
            algo_data = gpu_scaling[gpu_scaling['algo'].str.contains(algo, na=False)]
            if len(algo_data) > 0:
                algo_data = algo_data.sort_values('T_extracted')
                ax3.plot(algo_data['T_extracted'], algo_data['gpu_mem_MB'],
                        'o-', linewidth=2, markersize=6, label=algo.replace('_', ' ').title())
    
    ax3.set_xlabel('Sequence Length T')
    ax3.set_ylabel('GPU Memory Used (MB)')
    ax3.set_title('GPU Memory Usage vs T')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Bottom Right: Scaling Efficiency ===
    ax4 = axes[1, 1]
    
    # Efficiency = (T / time) normalized to T=min
    for algo in ['forward', 'viterbi', 'baum_welch_10']:
        algo_data = gpu_scaling[gpu_scaling['algo'].str.contains(algo, na=False)]
        if len(algo_data) > 1:
            algo_data = algo_data.sort_values('T_extracted')
            T_vals = algo_data['T_extracted'].values
            time_vals = algo_data['time_ms'].values
            
            # Throughput: elements processed per ms
            throughput = T_vals / time_vals
            # Normalize to first point
            efficiency = throughput / throughput[0]
            
            ax4.plot(T_vals, efficiency, 'o-', linewidth=2, markersize=6,
                    label=algo.replace('_', ' ').title())
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal (linear)')
    ax4.set_xlabel('Sequence Length T')
    ax4.set_ylabel('Relative Efficiency')
    ax4.set_title('Scaling Efficiency (Higher = Better)')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved: {output_path}")


# =============================================================================
# GRAPH 5: CPU Cache Efficiency
# =============================================================================

def plot_cpu_cache_metrics(cpu_df: Optional[pd.DataFrame],
                           output_path: str = 'cpu_cache_metrics.png'):
    """
    Plot CPU cache efficiency metrics.
    """
    if cpu_df is None:
        print("⚠ No CPU data available for cache plot")
        return
    
    # Check if cache metrics are available
    has_cache = 'cache_misses' in cpu_df.columns and cpu_df['cache_misses'].sum() > 0
    
    if not has_cache:
        # Create placeholder figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 
                'Cache metrics not available.\n\n'
                'To collect cache data, run with perf:\n'
                'perf stat -e cache-references,cache-misses,L1-dcache-load-misses \\\n'
                '         ./test_profile_cpu_robust data/bench results/cpu',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('CPU Cache Metrics')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Saved: {output_path} (placeholder)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cpu_scaling = filter_scaling_T(cpu_df)
    
    # === Top Left: Cache Miss Rate vs T ===
    ax1 = axes[0, 0]
    
    for algo in ['forward', 'viterbi', 'baum_welch_10']:
        algo_data = cpu_scaling[cpu_scaling['algo'].str.contains(algo, na=False)]
        if len(algo_data) > 0:
            algo_data = algo_data.sort_values('T_extracted')
            ax1.plot(algo_data['T_extracted'], algo_data['cache_miss_rate'] * 100,
                    'o-', linewidth=2, markersize=6, label=algo)
    
    ax1.set_xlabel('Sequence Length T')
    ax1.set_ylabel('Cache Miss Rate (%)')
    ax1.set_title('Cache Miss Rate vs T')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Top Right: IPC (Instructions per Cycle) ===
    ax2 = axes[0, 1]
    
    for algo in ['forward', 'viterbi', 'baum_welch_10']:
        algo_data = cpu_scaling[cpu_scaling['algo'].str.contains(algo, na=False)]
        if len(algo_data) > 0:
            algo_data = algo_data.sort_values('T_extracted')
            ax2.plot(algo_data['T_extracted'], algo_data['ipc'],
                    'o-', linewidth=2, markersize=6, label=algo)
    
    ax2.set_xlabel('Sequence Length T')
    ax2.set_ylabel('IPC (Instructions per Cycle)')
    ax2.set_title('CPU IPC vs T')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Bottom: Memory Usage ===
    ax3 = axes[1, 0]
    
    for algo in ['forward', 'viterbi', 'baum_welch_10']:
        algo_data = cpu_scaling[cpu_scaling['algo'].str.contains(algo, na=False)]
        if len(algo_data) > 0:
            algo_data = algo_data.sort_values('T_extracted')
            ax3.plot(algo_data['T_extracted'], algo_data['memory_kb'] / 1024,
                    'o-', linewidth=2, markersize=6, label=algo)
    
    ax3.set_xlabel('Sequence Length T')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('CPU Memory Usage vs T')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Bottom Right: Cache Misses (absolute) ===
    ax4 = axes[1, 1]
    
    algos = cpu_df['algo'].unique()[:5]  # Limit to 5 algos
    L1_misses = [cpu_df[cpu_df['algo'] == a]['L1_dcache_misses'].mean() for a in algos]
    LLC_misses = [cpu_df[cpu_df['algo'] == a]['LLC_misses'].mean() for a in algos]
    
    x = np.arange(len(algos))
    width = 0.35
    
    ax4.bar(x - width/2, np.array(L1_misses)/1e6, width, label='L1 Misses (M)', alpha=0.8)
    ax4.bar(x + width/2, np.array(LLC_misses)/1e6, width, label='LLC Misses (M)', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(algos, rotation=45, ha='right')
    ax4.set_ylabel('Cache Misses (Millions)')
    ax4.set_title('Cache Misses by Algorithm')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved: {output_path}")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(cpu_df: Optional[pd.DataFrame],
                           gpu_df: Optional[pd.DataFrame],
                           hmmlearn_df: Optional[pd.DataFrame],
                           output_path: str = 'benchmark_summary.txt'):
    """Generate text summary report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append(" " * 20 + "BENCHMARK SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # CPU Summary
    if cpu_df is not None:
        lines.append("CPU PROFILING (Hassan et al. Implementation)")
        lines.append("-" * 70)
        lines.append(f"  Total tests: {len(cpu_df)}")
        lines.append(f"  Algorithms: {', '.join(cpu_df['algo'].unique())}")
        lines.append("")
        lines.append("  Best times (smallest T):")
        for algo in cpu_df['algo'].unique():
            data = cpu_df[cpu_df['algo'] == algo]
            min_time = data['time_ms'].min()
            lines.append(f"    {algo:30s}: {min_time:10.3f} ms")
        lines.append("")
    
    # GPU Summary
    if gpu_df is not None:
        lines.append("GPU PROFILING (Hassan et al. Implementation)")
        lines.append("-" * 70)
        lines.append(f"  Total tests: {len(gpu_df)}")
        if 'device' in gpu_df.columns:
            lines.append(f"  Device: {gpu_df['device'].iloc[0]}")
        lines.append("")
        lines.append("  Best times (smallest T):")
        for algo in gpu_df['algo'].unique():
            data = gpu_df[gpu_df['algo'] == algo]
            min_time = data['time_ms'].min()
            lines.append(f"    {algo:30s}: {min_time:10.3f} ms")
        lines.append("")
    
    # Speedup Summary
    if cpu_df is not None and gpu_df is not None:
        lines.append("SPEEDUP SUMMARY (GPU vs CPU)")
        lines.append("-" * 70)
        
        for algo in ['forward', 'viterbi', 'baum_welch_10']:
            cpu_data = cpu_df[cpu_df['algo'].str.contains(algo, na=False)]
            gpu_data = gpu_df[gpu_df['algo'].str.contains(algo, na=False)]
            
            if len(cpu_data) > 0 and len(gpu_data) > 0:
                max_speedup = (cpu_data['time_ms'].max() / 
                              gpu_data.loc[gpu_data['T'] == cpu_data['T'].max(), 'time_ms'].values[0]
                              if len(gpu_data.loc[gpu_data['T'] == cpu_data['T'].max()]) > 0 else 0)
                lines.append(f"  {algo:25s}: up to {max_speedup:6.2f}x speedup")
        lines.append("")
    
    # hmmlearn Summary
    if hmmlearn_df is not None:
        lines.append("HMMLEARN BASELINE")
        lines.append("-" * 70)
        lines.append(f"  Total tests: {len(hmmlearn_df)}")
        lines.append("")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Report saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze HMM benchmark results and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_benchmark_results.py --cpu-csv results/cpu_results.csv
  python analyze_benchmark_results.py --cpu-csv results/cpu.csv --gpu-csv results/gpu.csv
  python analyze_benchmark_results.py --all-csv results/  # Auto-detect files in directory
        """
    )
    
    parser.add_argument('--cpu-csv', help='Path to CPU results CSV')
    parser.add_argument('--gpu-csv', help='Path to GPU results CSV')
    parser.add_argument('--hmmlearn-csv', help='Path to hmmlearn results CSV')
    parser.add_argument('--output-dir', default='results/figures',
                        help='Output directory for figures (default: results/figures)')
    
    args = parser.parse_args()
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         BENCHMARK ANALYSIS - HMM CPU vs GPU vs hmmlearn      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Load data
    cpu_df, gpu_df, hmmlearn_df = load_results(args.cpu_csv, args.gpu_csv, args.hmmlearn_csv)
    
    if cpu_df is None and gpu_df is None and hmmlearn_df is None:
        print("❌ No data found! Please provide at least one CSV file.")
        print("   Run profiling first:")
        print("     ./test_profile_cpu_robust data/bench results/cpu")
        print("     ./test_profile_gpu_robust data/bench results/gpu")
        print("     python benchmark_hmmlearn.py data/bench results/hmmlearn")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating figures...\n")
    
    # Generate all plots
    plot_time_vs_T(cpu_df, gpu_df, hmmlearn_df, 
                   output_path=str(output_dir / 'time_vs_T.png'))
    
    plot_speedup(cpu_df, gpu_df, hmmlearn_df,
                 output_path=str(output_dir / 'speedup_vs_T.png'))
    
    plot_convergence(cpu_df, gpu_df, hmmlearn_df,
                     output_path=str(output_dir / 'convergence.png'))
    
    plot_gpu_metrics(gpu_df,
                     output_path=str(output_dir / 'gpu_metrics.png'))
    
    plot_cpu_cache_metrics(cpu_df,
                           output_path=str(output_dir / 'cpu_cache_metrics.png'))
    
    # Generate summary report
    print("\n📝 Generating summary report...\n")
    generate_summary_report(cpu_df, gpu_df, hmmlearn_df,
                           output_path=str(output_dir / 'benchmark_summary.txt'))
    
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    ANALYSIS COMPLETE                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nFigures saved in: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()