import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(y_true, y_pred, labels=None):
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    # Micro
    micro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0
    )
    # Macro
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    return {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'micro': {
            'precision': micro[0],
            'recall': micro[1],
            'f1': micro[2],
            'support': micro[3]
        },
        'macro': {
            'precision': macro[0],
            'recall': macro[1],
            'f1': macro[2],
            'support': macro[3]
        }
    }
