import sys
sys.path.insert(0, '.')
from uchi.tabular import TabularPredictor
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

def test_replay():
    def load_synthetic_tabular(n_samples=2000, n_features=10, noise=0.2):
        return make_classification(n_samples=n_samples, n_features=n_features, flip_y=noise)
        
    X, y = load_synthetic_tabular(n_samples=2000, n_features=10, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
