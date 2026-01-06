#!/usr/bin/env python3
"""
analyze_profiling.py
Script d'analyse et visualisation des résultats de profiling CPU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

import sys
import os

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_results(csv_path="build/profiling_results.csv"):
    """Charge les résultats du profiling"""
    if not Path(csv_path).exists():
        print(f"❌ Erreur: {csv_path} introuvable")
        print("💡 Exécutez d'abord: make profile-cpu-simple")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"✅ Chargé {len(df)} résultats depuis {csv_path}")
    return df

def plot_scaling_by_algo(df, output_dir="results/profiling_results"):
    """Graphiques de scaling par algorithme"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    algos = df['algo'].unique()
    
    # 1. Scaling en K (observations)
    plt.figure(figsize=(12, 8))
    for algo in algos:
        subset = df[df['algo'] == algo].sort_values('K')
        plt.plot(subset['K'], subset['time_ms'], marker='o', label=algo, linewidth=2)
    
    plt.xlabel('K (Dimensions observations)', fontsize=12)
    plt.ylabel('Temps (ms)', fontsize=12)
    plt.title('Scaling en fonction de K', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_K.png', dpi=150)
    print(f"📊 Généré: {output_dir}/scaling_K.png")
    plt.close()
    
    # 2. Scaling en T (timesteps)
    plt.figure(figsize=(12, 8))
    for algo in algos:
        subset = df[df['algo'] == algo].sort_values('T')
        if len(subset) > 1:
            plt.plot(subset['T'], subset['time_ms'], marker='s', label=algo, linewidth=2)
    
    plt.xlabel('T (Timesteps)', fontsize=12)
    plt.ylabel('Temps (ms)', fontsize=12)
    plt.title('Scaling en fonction de T', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_T.png', dpi=150)
    print(f"📊 Généré: {output_dir}/scaling_T.png")
    plt.close()
    
    # 3. Scaling en N (états)
    plt.figure(figsize=(12, 8))
    for algo in algos:
        subset = df[df['algo'] == algo].sort_values('N')
        if len(subset) > 1:
            plt.plot(subset['N'], subset['time_ms'], marker='^', label=algo, linewidth=2)
    
    plt.xlabel('N (Nombre d\'états)', fontsize=12)
    plt.ylabel('Temps (ms)', fontsize=12)
    plt.title('Scaling en fonction de N', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_N.png', dpi=150)
    print(f"📊 Généré: {output_dir}/scaling_N.png")
    plt.close()

def plot_heatmap_time(df, output_dir="results/profiling_results"):
    """Heatmap du temps d'exécution par (N, T)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    algos_to_plot = ['forward', 'backward', 'viterbi', 'smoothing']
    
    for algo in algos_to_plot:
        subset = df[df['algo'] == algo]
        if len(subset) == 0:
            continue
        
        pivot = subset.pivot_table(values='time_ms', index='N', columns='T', aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(label='Temps (ms)')
        plt.xlabel('T (Timesteps)', fontsize=12)
        plt.ylabel('N (États)', fontsize=12)
        plt.title(f'Heatmap temps - {algo.capitalize()}', fontsize=14, fontweight='bold')
        
        # Annotations
        xticks = range(len(pivot.columns))
        yticks = range(len(pivot.index))
        plt.xticks(xticks, pivot.columns, rotation=45)
        plt.yticks(yticks, pivot.index)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/heatmap_{algo}.png', dpi=150)
        print(f"📊 Généré: {output_dir}/heatmap_{algo}.png")
        plt.close()

def analyze_complexity(df):
    """Analyse de la complexité observée vs théorique"""
    print("\n" + "="*60)
    print("ANALYSE DE COMPLEXITÉ")
    print("="*60)
    
    # Cholesky : O(K^3)
    chol = df[df['algo'] == 'cholesky'].sort_values('K')
    if len(chol) > 3:
        ratio = chol['time_ms'] / (chol['K'] ** 3)
        avg_ratio = ratio.mean()
        std_ratio = ratio.std()
        print(f"\n🔹 Cholesky (O(K³)):")
        print(f"   Ratio T/K³ moyen: {avg_ratio:.2e} ± {std_ratio:.2e}")
        print(f"   Cohérence: {'✅ Bon' if std_ratio/avg_ratio < 0.3 else '⚠️ Variable'}")
    
    # Forward/Backward : O(T*N^2)
    for algo_name in ['forward', 'backward']:
        subset = df[df['algo'] == algo_name]
        if len(subset) > 3:
            ratio = subset['time_ms'] / (subset['T'] * subset['N']**2)
            avg_ratio = ratio.mean()
            std_ratio = ratio.std()
            print(f"\n🔹 {algo_name.capitalize()} (O(T·N²)):")
            print(f"   Ratio T/(T·N²) moyen: {avg_ratio:.2e} ± {std_ratio:.2e}")
            print(f"   Cohérence: {'✅ Bon' if std_ratio/avg_ratio < 0.3 else '⚠️ Variable'}")

def compare_algos(df):
    """Comparaison relative des algos"""
    print("\n" + "="*60)
    print("COMPARAISON RELATIVE DES ALGORITHMES")
    print("="*60)
    
    # Prendre une config médiane
    median_config = df.groupby('config').size().idxmax()
    subset = df[df['config'] == median_config]
    
    print(f"\nConfiguration: {median_config}")
    print(f"N={subset['N'].iloc[0]}, T={subset['T'].iloc[0]}, K={subset['K'].iloc[0]}")
    print("\n" + "-"*60)
    
    for _, row in subset.sort_values('time_ms', ascending=False).iterrows():
        bar_length = int(row['time_ms'] / subset['time_ms'].max() * 40)
        bar = '█' * bar_length
        print(f"{row['algo']:20s} {bar:40s} {row['time_ms']:8.2f} ms")

def generate_summary_report(df, output_dir="build/profiling_results"):
    """Génère un rapport texte résumé"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = f"{output_dir}/summary_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" " * 15 + "RAPPORT PROFILING CPU - HMM\n")
        f.write("="*70 + "\n\n")
        
        # Statistiques globales
        f.write("📊 STATISTIQUES GLOBALES\n")
        f.write("-"*70 + "\n")
        f.write(f"Nombre de tests   : {len(df)}\n")
        f.write(f"Algorithmes testés: {', '.join(df['algo'].unique())}\n")
        f.write(f"Configurations    : {df['config'].nunique()}\n\n")
        
        # Par algorithme
        f.write("⏱️  TEMPS MOYEN PAR ALGORITHME\n")
        f.write("-"*70 + "\n")
        for algo in df['algo'].unique():
            subset = df[df['algo'] == algo]
            f.write(f"{algo:20s}: {subset['time_ms'].mean():10.2f} ms "
                   f"(min: {subset['time_ms'].min():.2f}, max: {subset['time_ms'].max():.2f})\n")
        
        f.write("\n💾 MÉMOIRE ESTIMÉE PAR CONFIGURATION\n")
        f.write("-"*70 + "\n")
        for config in df['config'].unique():
            subset = df[df['config'] == config]
            mem = subset['memory_mb'].iloc[0]
            f.write(f"{config:20s}: {mem:6d} MB\n")
        
        f.write("\n🔥 TOP 5 TESTS LES PLUS LENTS\n")
        f.write("-"*70 + "\n")
        top5 = df.nlargest(5, 'time_ms')
        for _, row in top5.iterrows():
            f.write(f"{row['algo']:15s} {row['config']:15s} "
                   f"N={row['N']:3d} T={row['T']:5d} K={row['K']:5d} "
                   f"=> {row['time_ms']:10.2f} ms\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"📄 Rapport généré: {report_path}")
    
    # Afficher le rapport
    with open(report_path, 'r') as f:
        print("\n" + f.read())

def main():
    """Point d'entrée principal"""
    print("\n" + "="*70)
    print(" " * 20 + "ANALYSE PROFILING CPU")
    print("="*70 + "\n")
    
    # Charger données
    df = load_results()
    
    # Générer visualisations
    print("\n📊 Génération des visualisations...")
    plot_scaling_by_algo(df)
    plot_heatmap_time(df)
    
    # Analyses
    analyze_complexity(df)
    compare_algos(df)
    
    # Rapport final
    generate_summary_report(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE")
    print("="*70)
    print("\n💡 Fichiers générés dans build/profiling_results/")
    print("   - scaling_*.png : Graphiques de scaling")
    print("   - heatmap_*.png : Heatmaps temps d'exécution")
    print("   - summary_report.txt : Rapport texte complet\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analyse interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)