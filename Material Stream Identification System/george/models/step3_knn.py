import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

X_train, y_train = joblib.load("../../dataset/features/train_hog_hsv_features.pkl")

X_test, y_test = joblib.load("../../dataset/features/test_hog_hsv_features.pkl")

scaler = joblib.load("../storage/scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_tset_scaled = scaler.transform(X_test)


pca = PCA(n_components=50, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_tset_scaled)

knn = KNeighborsClassifier(
    n_neighbors=7,
    metric="euclidean",  # standard distance
    weights="distance",  # closer neighbors matter more
)

knn.fit(X_train_pca, y_train)

y_test_pred = knn.predict(X_test_pca)
result = accuracy_score(y_test, y_test_pred)
print("Validation accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

joblib.dump(knn, "../storage/knn.pkl")
