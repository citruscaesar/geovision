from typing import Optional
from numpy.typing import NDArray

import numpy as np
import pandas as pd

# TODO: add support for multilabel classification metrics

def get_classification_metrics_dict(conf_mat: NDArray, class_names: Optional[tuple[str]] = None) -> dict[str, list]:
    num_classes = conf_mat.shape[0]
    num_samples = conf_mat.sum()
    class_names = class_names or tuple(f"class_{i:02}" for i in range(num_classes))
    assert len(class_names) == num_classes, f"shape mismatch, expected an array of len = {num_classes}, got len = {len(class_names)}"
    
    metrics = {k:list() for k in ("precision", "recall", "f1", "iou")}
    metrics["class_names"] = class_names
    metrics["accuracy"] = conf_mat.trace() / num_samples
    metrics["support"] = conf_mat.sum(axis = 0).tolist()
    for c in range(num_classes):
        tp, p, p_hat = conf_mat[c, c], conf_mat[c, :].sum(), conf_mat[:, c].sum()
        metrics["precision"].append((tp / p_hat) if p_hat > 0 else 0)
        metrics["recall"].append((tp / p) if p > 0 else 0)
        metrics["iou"].append(tp / (p+p_hat-tp) if (p+p_hat-tp) > 0 else 0)
        metrics["f1"].append( (2*tp) / (p+p_hat) if (p+p_hat) > 0 else 0)

    sup = metrics["support"] / num_samples 
    metrics["weighted"] = [np.dot(metrics[m], sup) for m in ("precision", "recall", "f1", "iou")]
    return metrics

def get_classification_metrics_df(metrics_dict: dict[str, list]):
    num_samples = sum(metrics_dict["support"])
    acc = metrics_dict["accuracy"]

    df = pd.DataFrame({k:metrics_dict[k] for k in ("class_names", "precision", "recall", "f1", "iou", "support")})
    df.loc[len(df.index)] = ["macro", df["precision"].mean(), df["recall"].mean(), df["f1"].mean(), df["iou"].mean(), num_samples]
    df.loc[len(df.index)] = ["weighted", *metrics_dict["weighted"], num_samples]
    df.loc[len(df.index)] = ["micro", acc, acc, acc, acc, num_samples]
    df = df.set_index("class_names")
    return df

def get_classwise_metrics_dict(conf_mat: NDArray, class_names: Optional[tuple[str]] = None, prefix: Optional[str] = "train/") -> dict[str, float]:
    metrics_dict = get_classification_metrics_dict(conf_mat, class_names)
    classwise_metrics = dict()
    for idx, class_name in enumerate(metrics_dict["class_names"]):
        for metric in ("precision", "recall", "f1", "iou"):
            classwise_metrics[f"{prefix}{class_name}_{metric}"] = metrics_dict[metric][idx]
    return classwise_metrics
