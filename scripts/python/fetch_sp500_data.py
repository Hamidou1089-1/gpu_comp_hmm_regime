#!/usr/bin/env python3
"""
fetch_sp500_data.py
Récupère les données S&P500 et calcule les features pour le HMM de détection de régime.

Features:
1. Log Returns: log(P_t / P_{t-1})
2. Volatilité réalisée: std des returns sur fenêtre glissante
3. Volume normalisé: (V_t - mean(V)) / std(V)

Ces 3 features permettent de caractériser les régimes de marché:
- Bull: returns > 0, faible volatilité, volume modéré
- Bear: returns < 0, haute volatilité, volume élevé (panique)
- Sideways: returns ~ 0, faible volatilité, faible volume
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json
import struct
from pathlib import Path
from datetime import datetime
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# SEED GLOBAL POUR REPRODUCTIBILITÉ
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


def ensure_dirs():
    """Crée les répertoires nécessaires"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ACQUISITION DES DONNÉES
# ============================================================================

def fetch_sp500_data(
    ticker: str = "SPY",
    start_date: str = "2010-01-01",
    end_date: str = "2026-01-01",
    save_raw: bool = True
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV du S&P500 (via SPY ETF)
    
    Args:
        ticker: Symbole (SPY pour S&P500 ETF)
        start_date: Date de début
        end_date: Date de fin (None = aujourd'hui)
        save_raw: Sauvegarder les données brutes
        
    Returns:
        DataFrame avec OHLCV
    """
    print(f"📥 Téléchargement {ticker} depuis {start_date}...")
    
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=True  # Ajuste pour dividendes/splits
    )
    
    if data.empty:
        raise ValueError(f"Aucune donnée récupérée pour {ticker}")
    
    print(f"✓ {len(data)} jours de données ({data.index[0].date()} → {data.index[-1].date()})")
    
    if save_raw:
        raw_path = RAW_DIR / f"{ticker}_raw.csv"
        data.to_csv(raw_path)
        print(f"✓ Données brutes sauvegardées: {raw_path}")
    
    return data


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_features(
    data: pd.DataFrame,
    vol_window: int = 20,
    vol_annualize: bool = True
) -> pd.DataFrame:
    """
    Calcule les 3 features pour le HMM.
    
    Args:
        data: DataFrame OHLCV
        vol_window: Fenêtre pour volatilité réalisée (jours)
        vol_annualize: Annualiser la volatilité (×√252)
        
    Returns:
        DataFrame avec [returns, volatility, volume_norm]
    """
    print(f"\n📊 Calcul des features (fenêtre vol={vol_window} jours)...")
    
    # 1. Log Returns
    close = data['Close']
    returns = np.log(close / close.shift(1))
    
    # 2. Volatilité réalisée (rolling std des returns)
    volatility = returns.rolling(window=vol_window).std()
    if vol_annualize:
        volatility = volatility * np.sqrt(252)
    
    # 3. Volume normalisé (z-score)
    volume = data['Volume']
    volume_mean = volume.rolling(window=vol_window).mean()
    volume_std = volume.rolling(window=vol_window).std()
    volume_norm = (volume - volume_mean) / volume_std
    
    # Combine
    features = pd.DataFrame({
        'returns': returns.to_numpy().flatten(),
        'volatility': volatility.to_numpy().flatten(),
        'volume_norm': volume_norm.to_numpy().flatten()
    }, index=data.index)
    
    # Drop NaN (début de série dû aux fenêtres glissantes)
    features = features.dropna()
    
    print(f"✓ Features calculées: {len(features)} observations × {features.shape[1]} features")
    
    # Statistiques
    print("\n📈 Statistiques des features:")
    print(features.describe().round(6))
    
    return features


# ============================================================================
# EXPORT POUR C++
# ============================================================================

def export_for_cpp(
    features: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
    prefix: str = "sp500"
):
    """
    Exporte les données en format binaire pour C++.
    
    Format:
    - {prefix}_observations.bin: float32 array [T × K], row-major
    - {prefix}_info.json: métadonnées (T, K, dates, etc.)
    """
    T, K = features.shape
    
    # 1. Export binaire (float32, row-major)
    data_flat = features.values.astype(np.float32).flatten(order='C')
    
    bin_path = output_dir / f"{prefix}_observations.bin"
    data_flat.tofile(bin_path)
    
    print(f"\n💾 Export binaire: {bin_path}")
    print(f"   Taille: {bin_path.stat().st_size / 1024:.1f} KB")
    print(f"   Shape: [{T} × {K}] = {T * K} floats")
    
    # 2. Export métadonnées JSON
    info = {
        'T': int(T),
        'K': int(K),
        'dtype': 'float32',
        'order': 'row-major (C)',
        'features': list(features.columns),
        'date_start': str(features.index[0].date()),
        'date_end': str(features.index[-1].date()),
        'seed': GLOBAL_SEED,
        'created_at': datetime.now().isoformat()
    }
    
    json_path = output_dir / f"{prefix}_info.json"
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Métadonnées: {json_path}")
    
    # 3. Export CSV (pour inspection/debug)
    csv_path = output_dir / f"{prefix}_features.csv"
    features.to_csv(csv_path)
    print(f"✓ CSV (debug): {csv_path}")
    
    # 4. Export des dimensions dans un fichier simple pour C++ facile
    dims_path = output_dir / f"{prefix}_dims.txt"
    with open(dims_path, 'w') as f:
        f.write(f"{T} {K}\n")
    print(f"✓ Dimensions: {dims_path}")
    
    return bin_path, json_path


def export_init_params(
    T: int, K: int, N: int = 3,
    output_dir: Path = PROCESSED_DIR,
    prefix: str = "sp500"
):
    """
    Génère et exporte les paramètres initiaux du HMM pour C++.
    
    Args:
        T, K: Dimensions des données
        N: Nombre d'états cachés (3 = Bull/Bear/Sideways)
    """
    print(f"\n🔧 Génération des paramètres initiaux (N={N} états)...")
    
    # SEED pour reproductibilité
    np.random.seed(GLOBAL_SEED)
    
    # 1. Distribution initiale π (uniforme)
    pi = np.full(N, 1.0 / N, dtype=np.float32)
    
    # 2. Matrice de transition A (persistante)
    # Les régimes ont tendance à persister
    persistence = 0.9
    A = np.full((N, N), (1 - persistence) / (N - 1), dtype=np.float32)
    np.fill_diagonal(A, persistence)
    
    # 3. Moyennes μ (séparées pour les 3 régimes)
    # Bull: returns positifs, vol basse, volume normal
    # Bear: returns négatifs, vol haute, volume élevé
    # Sideways: returns ~0, vol basse, volume bas
    mu = np.array([
        [0.001, 0.10, 0.0],    # Bull
        [-0.002, 0.30, 1.0],   # Bear
        [0.0, 0.15, -0.5],     # Sideways
    ], dtype=np.float32)
    
    # 4. Covariances Σ (diagonal avec petite corrélation)
    base_var = np.array([0.0002, 0.01, 0.5], dtype=np.float32)  # Par feature
    Sigma = np.zeros((N, K, K), dtype=np.float32)
    for i in range(N):
        # Matrice diagonale + régularisation
        Sigma[i] = np.diag(base_var) * (1.0 + 0.2 * i)  # Plus de variance dans Bear
        # Régularisation pour garantir positive-définite
        Sigma[i] += np.eye(K, dtype=np.float32) * 1e-4
    
    # Export
    paths = {}
    
    paths['pi'] = output_dir / f"{prefix}_pi_init.bin"
    pi.tofile(paths['pi'])
    
    paths['A'] = output_dir / f"{prefix}_A_init.bin"
    A.tofile(paths['A'])
    
    paths['mu'] = output_dir / f"{prefix}_mu_init.bin"
    mu.tofile(paths['mu'])
    
    paths['Sigma'] = output_dir / f"{prefix}_Sigma_init.bin"
    Sigma.tofile(paths['Sigma'])
    
    # JSON avec info
    params_info = {
        'N': int(N),
        'K': int(K),
        'seed': GLOBAL_SEED,
        'persistence': float(persistence),
        'files': {k: str(v) for k, v in paths.items()}
    }
    
    params_json_path = output_dir / f"{prefix}_params_info.json"
    with open(params_json_path, 'w') as f:
        json.dump(params_info, f, indent=2)
    
    print(f"✓ Paramètres exportés:")
    for name, path in paths.items():
        print(f"   {name}: {path}")
    
    return paths


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prépare les données S&P500 pour HMM')
    parser.add_argument('--ticker', default='SPY', help='Ticker (default: SPY)')
    parser.add_argument('--start', default='2005-01-01', help='Date début')
    parser.add_argument('--end', default=None, help='Date fin (default: now)')
    parser.add_argument('--vol-window', type=int, default=10, help='Fenêtre volatilité')
    parser.add_argument('--n-states', type=int, default=3, help='Nombre d\'états HMM')
    parser.add_argument('--prefix', default='sp500', help='Préfixe fichiers output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  S&P500 DATA PREPARATION FOR HMM")
    print("=" * 60)
    print(f"  Seed: {GLOBAL_SEED}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Period: {args.start} → {args.end or 'now'}")
    print("=" * 60)
    
    ensure_dirs()
    
    # 1. Fetch data
    raw_data = fetch_sp500_data(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end
    )
    
    # 2. Compute features
    features = compute_features(raw_data, vol_window=args.vol_window)
    
    # 3. Export for C++
    export_for_cpp(features, prefix=args.prefix)
    
    # 4. Generate initial params
    T, K = features.shape
    export_init_params(T, K, N=args.n_states, prefix=args.prefix)
    
    print("\n" + "=" * 60)
    print("  ✅ DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {PROCESSED_DIR}")
    print(f"  Ready for C++ profiling!\n")


if __name__ == '__main__':
    main()