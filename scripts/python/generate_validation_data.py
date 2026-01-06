import numpy as np
import yfinance as yf
from hmmlearn import hmm
import struct
import os

# ==========================================
# 1. TÉLÉCHARGEMENT & FEATURES
# ==========================================
print("Downloading Data...")
data = yf.download('SPY', start='2020-01-01', end='2023-01-01')

# Feature 1: Log Returns
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
# Feature 2: Volatilité (Range) - Proxy simple
vol = np.log(data['High'] / data['Low']).dropna()

# Alignement
df = returns.to_frame(name='Ret').join(vol.to_frame(name='Vol')).dropna()
X = df.values # [T, K]

T, K = X.shape
N = 3 # Nombre d'états (Bull, Bear, Sideways)

print(f"Data shape: T={T}, K={K}")

# ==========================================
# 2. INITIALISATION DU MODÈLE (HMMLearn)
# ==========================================
# covariance_type='full' est important car ton C++ gère des matrices K*K
model = hmm.GaussianHMM(n_components=N, covariance_type='full', n_iter=1, tol=1e-4, init_params="")

# Init manuelle pour être sûr de ce qu'on envoie au C++
np.random.seed(42)
model.startprob_ = np.full(N, 1/N)
# Transition un peu collante (diagonale forte)
model.transmat_ = np.array([[0.9, 0.05, 0.05], 
                            [0.05, 0.9, 0.05], 
                            [0.05, 0.05, 0.9]])
# Moyennes aléatoires autour de 0
model.means_ = np.random.normal(0, 0.01, (N, K))
# Covariances : Identité un peu bruitée
model.covars_ = np.tile(np.eye(K), (N, 1, 1)) * 0.001

# ==========================================
# 3. EXPORT BINAIRE POUR C++
# ==========================================
def write_bin(filename, array):
    with open(filename, 'wb') as f:
        # Flatten row-major
        f.write(array.astype(np.float32).tobytes())
    print(f"Saved {filename}")

os.makedirs("data_cpp", exist_ok=True)
write_bin("data_cpp/obs.bin", X)
write_bin("data_cpp/pi_init.bin", model.startprob_)
write_bin("data_cpp/A_init.bin", model.transmat_)
write_bin("data_cpp/mu_init.bin", model.means_)
write_bin("data_cpp/sigma_init.bin", model.covars_)

# Sauvegarde des dimensions dans un txt
with open("data_cpp/dims.txt", "w") as f:
    f.write(f"{T} {N} {K}")

# ==========================================
# 4. REFERENCE RUN (1 STEP)
# ==========================================
# On triche un peu : hmmlearn fait tout en interne. 
# Pour avoir la LL après 1 step, on fit.
print("\n--- Running HMMLEARN Reference ---")
model.fit(X)

print(f"Initial Score (LogLikelihood): {model.score(X)}")
print("Note: HMMLearn score() est la LL finale après fit.")

# On affiche les params APRES 1 step pour comparer
print("\nExpected Mu (State 0) after 1 iter:")
print(model.means_[0])