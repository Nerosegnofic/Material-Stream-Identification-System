import os
import joblib
import numpy as np
from models.knn_model import create_knn

SEED = 42

def main():
    train_data_path = "../data_preparation/train_features.npz"
    test_data_path = "../data_preparation/test_features.npz"

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print(f"Training samples: {len(y_train)}, feature dim: {X_train.shape[1]}")
    print(f"Test samples:     {len(y_test)}, feature dim: {X_test.shape[1]}")

    knn_model = create_knn()
    knn_model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(knn_model, "saved_models/knn.pkl")
    np.savez("saved_models/test_set.npz", X=X_test, y=y_test)

    print("KNN trained successfully using CNN features")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples:  {len(y_test)}")

if __name__ == "__main__":
    main()
