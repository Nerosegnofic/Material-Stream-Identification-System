import joblib
from sklearn.preprocessing import StandardScaler
import os

X, y = joblib.load(os.path.join(os.path.dirname(__file__), "../..", "dataset/features/train_hog_hsv_features.pkl"))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
os.makedirs((os.path.join(os.path.dirname(__file__), "..", "storage")), exist_ok=True)
joblib.dump(scaler, (os.path.join(os.path.dirname(__file__), "..", "storage/scaler.pkl")))