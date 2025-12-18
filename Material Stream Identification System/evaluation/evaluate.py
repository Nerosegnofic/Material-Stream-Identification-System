from sklearn.metrics import accuracy_score, classification_report
from utils.constants import CLASS_NAMES
from models.rejection import reject_unknown

def evaluate(model, X_test, y_test, threshold=0.6):
    probs = model.predict_proba(X_test)

    preds = []
    for p in probs:
        preds.append(reject_unknown(p, threshold))

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)
    print(classification_report(
        y_test,
        preds,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
        zero_division=0
    ))

    report = classification_report(
        y_test,
        preds,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
        output_dict=True,
        zero_division=0
    )

    return acc, report
