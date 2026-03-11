import logging
from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def evaluate_classification(y_true, y_pred, y_probs=None) -> Dict[str, Any]:
    """
    Standard evaluation metrics for classification tasks.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
    }

    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs)
            results["roc_auc"] = auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    try:
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
    except Exception as e:
        logger.warning(f"Could not calculate Confusion Matrix: {e}")

    return results
