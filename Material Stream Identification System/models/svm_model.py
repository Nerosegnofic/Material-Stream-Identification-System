from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_svm():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            decision_function_shape="ovr"
        ))
    ])

    return model
