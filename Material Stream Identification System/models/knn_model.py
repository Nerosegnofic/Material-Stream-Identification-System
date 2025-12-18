from sklearn.neighbors import KNeighborsClassifier

def create_knn(k=5):
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance'
    )
    return model
