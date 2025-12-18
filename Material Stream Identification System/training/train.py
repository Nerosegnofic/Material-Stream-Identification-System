import os
import joblib
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models.svm_model import create_svm

if __name__ == "__main__":

    npz_path = os.path.join("..", "data_preparation", "features.npz")
    data = np.load(npz_path)
    X, y = data["X"], data["y"]

    X, y = shuffle(X, y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = create_svm()
    model.fit(X_train, y_train)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/svm.pkl")
    np.savez("saved_models/test_set.npz", X=X_test, y=y_test)

    print("SVM trained successfully")
    print(f"Train samples: {len(y_train)}")
    print(f"Test samples:  {len(y_test)}")
