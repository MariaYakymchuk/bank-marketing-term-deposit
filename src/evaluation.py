"""Threshold choice and error analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def find_best_threshold(y_true, proba, thresholds=np.arange(0.05, 0.8, 0.01)):
    """Return the threshold with the best F1 score and this F1 score."""
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    best = int(np.argmax(f1s))
    return round(float(thresholds[best]), 2), f1s[best]


def label_errors(X: pd.DataFrame, y_true, proba, threshold: float) -> pd.DataFrame:
    """Add real answer, probability, prediction and the result group (TP / TN / FN / FP) to X."""
    df = X.copy()
    df['real'] = np.asarray(y_true)
    df['proba'] = proba
    df['pred'] = (proba >= threshold).astype(int)

    groups = {
        (1, 1): 'TP (correct yes)',
        (0, 0): 'TN (correct no)',
        (1, 0): 'FN (missed yes)',
        (0, 1): 'FP (false alarm)',
    }
    df['group'] = [groups[(r, p)] for r, p in zip(df['real'], df['pred'])]
    return df
