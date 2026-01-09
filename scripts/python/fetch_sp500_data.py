import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
from hmmlearn import hmm

# Configuration
DATA_DIR = "../../data/finance"
RESULTS_DIR = "../../results/finance"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def fetch_and_prepare_data():
    print("📥 Downloading S&P 500 Data (2000-2024)...")
    # Téléchargement
    df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31")
    df = pd.DataFrame(df['Close'])
    df.columns = ['price']
    
    # Feature Engineering
    # 1. Log Returns
    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
    
    # 2. Volatilité Court Terme (10 jours)
    df['vol_short'] = df['log_ret'].rolling(window=10).std()
    
    # 3. Volatilité Long Terme (3 mois = 63 jours de trading)
    df['vol_long'] = df['log_ret'].rolling(window=63).std()
    
    # Nettoyage NaN
    df.dropna(inplace=True)
    
    # --- NORMALISATION (CRITIQUE pour HMM Gaussien) ---
    # On stocke moyenne et std pour dé-normaliser si besoin
    features = ['log_ret', 'vol_short', 'vol_long']
    X = df[features].values.astype(np.float32)
    
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_scaled = (X - means) / stds
    
    print(f"📊 Dataset ready: T={X.shape[0]}, K={X.shape[1]}")
    return df, X_scaled

def write_bin(filename, array):
    with open(filename, 'wb') as f:
        f.write(array.astype(np.float32).tobytes())

def export_to_bin(X, filename):
    # Format binaire simple (Row-Major float32)
    write_bin(filename, X)
    
    # Fichier dims
    base = os.path.splitext(filename)[0]
    with open(f"{base}_dims.txt", 'w') as f:
        f.write(f"{X.shape[0]} {3} {X.shape[1]}") # T N K (N=3 états par défaut)



def run_hmmlearn_benchmark(X, n_states=3):
    print("\n🐍 Running hmmlearn baseline...")
    start = time.time()
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, tol=1e-4, random_state=42)
    model.fit(X)
    train_time = time.time() - start
    
    start = time.time()
    states = model.predict(X)
    inference_time = time.time() - start
    
    print(f"   Train Time: {train_time*1000:.2f} ms")
    print(f"   Infer Time: {inference_time*1000:.2f} ms")
    return states

def plot_regimes(df, states): # , output_file):
    # --- RE-ORDER STATES ---
    # On veut que :
    # Etat 0 = Faible Volatilité (Bull) -> Vert
    # Etat 1 = Moyenne Volatilité (Correction) -> Orange/Gris
    # Etat 2 = Haute Volatilité (Crise) -> Rouge
    
    # On calcule la volatilité moyenne par état prédit
    df['state'] = states
    vol_by_state = df.groupby('state')['vol_short'].mean()
    
    # Mapping: trier les états par volatilité croissante
    sorted_states = vol_by_state.sort_values().index
    state_map = {old: new for new, old in enumerate(sorted_states)}
    df['mapped_state'] = df['state'].map(state_map)
    
    # Couleurs
    colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Vert, Jaune, Rouge
    labels = ['Bull / Low Vol', 'Correction / Med Vol', 'Crisis / High Vol']
    
    plt.figure(figsize=(15, 8))
    
    # On trace le prix log pour mieux voir les variations
    price = np.log(df['price'])
    
    
    for i in range(3):
        mask = (df['mapped_state'] == i)
        plt.fill_between(df.index, price.min(), price.max(), where=mask, 
                         color=colors[i], alpha=0.3, label=labels[i])
    
    plt.plot(df.index, price, color='black', linewidth=1)
    
    plt.title('S&P 500 Market Regimes Detection (GPU Accelerated HMM)', fontsize=16)
    plt.ylabel('Log Price')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # plt.savefig(output_file, dpi=150)
    # print(f"🖼️ Graph saved to {output_file}")

def main():
    # 1. Data
    df, X = fetch_and_prepare_data()
    bin_path = f"{DATA_DIR}/sp500_obs.bin"
    export_to_bin(X, bin_path)
    
    
    

if __name__ == "__main__":
    main()