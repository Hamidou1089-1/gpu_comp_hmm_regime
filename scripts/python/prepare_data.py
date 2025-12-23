"""
Simple script to download financial data and prepare for HMM
- Download closing prices from yfinance
- Calculate log returns
- Save as flat binary array for C++
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_tickers(n_total=140):
    """Get diverse tickers from multiple sectors"""
    print(f"Fetching {n_total} tickers from various sectors...")
    
    sectors = [
        'technology', 'energy', 'utilities', 'healthcare',
        'consumer-defensive', 'basic-materials', 'communication-services',
        'financial-services', 'industrials', 'real-estate'
    ]
    
    tickers = []
    per_sector = n_total // len(sectors) + 1
    
    for sector in sectors:
        try:
            top = yf.Sector(sector).top_companies
            tickers.extend(top.index.tolist()[:per_sector])
        except:
            pass
    
    return list(set(tickers))[:n_total]


def main():
    print("=" * 60)
    print("Financial Data Preparation for HMM")
    print("=" * 60)
    
    # 1. Get tickers
    tickers = get_tickers(n_total=140)
    print(f"Selected {len(tickers)} tickers\n")
    
    # 2. Download closing prices (15 years)
    print("Downloading 15 years of closing prices...")
    data = yf.download(tickers, period="15y", progress=True)
    prices = data['Close']  
    
    # Clean: remove tickers with too many NaN
    prices = prices.dropna(axis=1, thresh=int(0.95 * len(prices)))
    prices = prices.fillna(method='ffill').dropna()
    
    print(f"✓ Downloaded: T={len(prices)} days, K={len(prices.columns)} assets\n")
    
    # 3. Calculate log returns: log(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    T = len(log_returns)
    K = len(log_returns.columns)
    
    print(f"✓ Log returns: T={T}, K={K}")
    print(f"  Mean: {log_returns.mean().mean():.6f}")
    print(f"  Std:  {log_returns.std().mean():.6f}\n")
    
    # 4. Save as flat binary (row-major, float32)
    data_flat = log_returns.values.astype(np.float32).flatten(order='C')
    
    bin_path = PROCESSED_DIR / "observations.bin"
    data_flat.tofile(bin_path)
    
    print(f"✓ Saved binary: {bin_path}")
    print(f"  Size: {bin_path.stat().st_size / 1024:.1f} KB\n")
    
    # 5. Save metadata
    info = {
        'T': T,
        'K': K,
        'dtype': 'float32',
        'order': 'row-major',
        'tickers': log_returns.columns.tolist(),
        'date_start': str(log_returns.index[0].date()),
        'date_end': str(log_returns.index[-1].date()),
    }
    
    json_path = PROCESSED_DIR / "observations_info.json"
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Saved metadata: {json_path}")
    
    # Optional: save CSV for inspection
    csv_path = PROCESSED_DIR / "log_returns.csv"
    log_returns.to_csv(csv_path)
    print(f"✓ Saved CSV (optional): {csv_path}")
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)
    print(f"\nFiles for C++:")
    print(f"  {bin_path}")
    print(f"  {json_path}")


if __name__ == '__main__':
    main()