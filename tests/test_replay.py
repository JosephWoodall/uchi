import sys
sys.path.insert(0, '.')
from uchi.tabular import TabularPredictor
from tests._datasets import load_synthetic_tabular
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

X, y = load_synthetic_tabular(n_samples=2000, n_features=10, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Benchmark 1: Pure online (no replay buffer)
mp_pure = TabularPredictor(n_bins=5, n_epochs=1, n_orderings=3)
mp_pure._disc = mp_pure._disc.fit(X_train.tolist()) # Initialize discretizer
mp_pure._lenc = mp_pure._lenc.fit(y_train.tolist())
mp_pure._rng = __import__('random').Random(42)
mp_pure._orders = mp_pure._orders or [[i for i in range(10)] for _ in range(3)]
mp_pure._preds = [__import__('uchi.tabular', fromlist=['_make_predictor'])._make_predictor(10, 0.08, 6.05, 0.65) for _ in range(3)]

t0 = time.time()
for row, label in zip(X_train.tolist(), y_train.tolist()):
    # bypass partial_fit replay buffer to simulate pure online
    encoded = mp_pure._disc._encode_row(row)
    lt = mp_pure._label_token(label)
    for p, order in zip(mp_pure._preds, mp_pure._orders):
        p.observe(lt) # hacky
        
# Actually let's just use the real API
mp_no_replay = TabularPredictor(n_bins=5, n_epochs=1, n_orderings=3)
mp_no_replay._replay_batch_size = 1 # Forces pure online
mp_no_replay.partial_fit(X_train.tolist(), y_train.tolist())
acc_no_replay = accuracy_score(y_test, mp_no_replay.predict(X_test.tolist()))

mp_replay = TabularPredictor(n_bins=5, n_epochs=5, n_orderings=3)
mp_replay._replay_batch_size = 100 # Uses replay buffer
mp_replay.partial_fit(X_train.tolist(), y_train.tolist())
acc_replay = accuracy_score(y_test, mp_replay.predict(X_test.tolist()))

print(f"Online (No Replay Buffer) Accuracy: {acc_no_replay:.3f}")
print(f"Online (With Replay Buffer) Accuracy: {acc_replay:.3f}")
