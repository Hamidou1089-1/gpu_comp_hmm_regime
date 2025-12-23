# Python Scripts

Scripts Python pour le téléchargement, preprocessing et analyse des données.

## Installation

### 1. Créer l'environnement virtuel
```bash
cd scripts/python
bash setup_venv.sh
```

### 2. Activer l'environnement
```bash
source .venv/bin/activate
```

### 3. Vérifier l'installation
```bash
python -c "import yfinance; print('yfinance OK')"
```

## Utilisation

### Télécharger les données
```bash
python data_download.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-01
```

### Preprocessing
```bash
python data_preprocess.py --input data/raw/stocks.csv --output data/processed/stocks_processed.csv
```

### Visualisation des résultats
```bash
python analysis/plot_results.py --input results/benchmarks/
```

## Structure des données
```
data/
├── raw/
│   ├── tickers.txt           # Liste des tickers
│   └── stocks_YYYYMMDD.csv   # Données brutes
└── processed/
    ├── train.csv             # Données d'entraînement
    ├── test.csv              # Données de test
    └── metadata.json         # Métadonnées
```

## Notes

- L'environnement virtuel est dans `.venv/` (ignoré par git)
- Les données brutes sont ignorées par git (trop volumineuses)
- Seuls les scripts et metadata sont versionnés