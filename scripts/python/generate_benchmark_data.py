#!/usr/bin/env python3
"""
scripts/generate_benchmark_data.py
Génère les datasets binaires pour le benchmarking rigoureux.
"""
import numpy as np
import os
import struct

OUTPUT_DIR = "../../data/bench"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

def write_bin(filename, array):
    with open(filename, 'wb') as f:
        f.write(array.astype(np.float32).tobytes())

def generate_dataset(T, N, K, label):
    print(f"Generating {label}: T={T}, N={N}, K={K}")
    
    # Paramètres "Ground Truth" stables
    means = np.random.normal(0, 1, (N, K))
    covars = np.tile(np.eye(K), (N, 1, 1))
    
    # Génération Données
    X = np.random.normal(0, 1, (T, K)).astype(np.float32)
    # On ajoute un peu de structure (juste pour que ce ne soit pas du pur bruit)
    for t in range(T):
        state = t % N
        X[t] += means[state]

    # Normalisation (CRITIQUE pour HMM)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Initialisation "Mauvaise" (pour que l'algo ait du travail)
    pi_init = np.ones(N) / N
    A_init = np.ones((N, N)) / N
    mu_init = np.random.normal(0, 0.5, (N, K))
    sigma_init = np.tile(np.eye(K), (N, 1, 1))

    # Sauvegarde
    prefix = f"{OUTPUT_DIR}/{label}"
    write_bin(f"{prefix}_obs.bin", X)
    write_bin(f"{prefix}_pi.bin", pi_init)
    write_bin(f"{prefix}_A.bin", A_init)
    write_bin(f"{prefix}_mu.bin", mu_init)
    write_bin(f"{prefix}_sigma.bin", sigma_init)
    
    with open(f"{prefix}_dims.txt", "w") as f:
        f.write(f"{T} {N} {K}")

# 1. Scaling T (Impact de la taille de séquence -> Hassan vs Standard)
t = 100
while t <= 100100: # Décommentez 1M pour stress test
    generate_dataset(t, 2, 2, f"scaling_T_{t}")
    t += 1000


# 3. Validation (Petit dataset pour vérifier la LL)
#generate_dataset(2048, 3, 4, "validation_convergence")

print("✅ Data generation complete.")