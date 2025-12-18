import numpy as np

def reject_unknown(probs, threshold=0.6):
    if np.max(probs) < threshold:
        return 6
    return np.argmax(probs)
