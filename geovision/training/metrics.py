from typing import Literal
from numpy.typing import NDArray

import numpy as np
import pandas as pd

# TODO: refactor get_classification_metrics
# TODO: add support for multilabel classification metrics

def get_classification_metrics_df(confusion_matrix: NDArray, class_names: tuple) -> pd.DataFrame:
    num_classes = confusion_matrix.shape[0]
    num_samples = np.sum(confusion_matrix)

    # NOTE: If required, add additional metrics BEFORE the support column
    df = pd.DataFrame(columns = ["class_name", "precision", "recall", "iou", "f1", "support"])
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        p_hat = np.sum(confusion_matrix[:, c])
        p = np.sum(confusion_matrix[c, :])

        precision = (tp / p_hat) if p_hat > 0 else 0
        recall = (tp / p) if p > 0 else 0
        iou = tp / (p+p_hat-tp) if (p+p_hat-tp) > 0 else 0
        f1 =  (2*tp) / (p+p_hat) if (p+p_hat) > 0 else 0
        support = np.sum(p)

        df.loc[len(df.index)] = [class_names[c].lower(), precision, recall, iou, f1, support]
        
    accuracy = confusion_matrix.trace() / num_samples

    # NOTE weighted_metric = np.dot(metric, support) 
    weighted_metrics = np.matmul((df["support"] / df["support"].sum()).to_numpy(), 
                                df[["precision", "recall", "iou", "f1"]].to_numpy())
    df.loc[len(df.index)] = ["accuracy", accuracy, accuracy, accuracy, accuracy, num_samples]
    df.loc[len(df.index)] = ["macro", df["precision"].mean(), df["recall"].mean(), df["iou"].mean(), df["f1"].mean(), num_samples]
    df.loc[len(df.index)] = ["weighted", *weighted_metrics, num_samples]
    df.set_index("class_name", inplace = True)
    return df

def get_classification_metrics_dict(metrics_df: pd.DataFrame) -> dict[str, float]:
    metrics = dict()
    for class_name, row in metrics_df.iterrows(): # type: ignore
        if str(class_name) == "accuracy":
            metrics["accuracy"] = row["precision"]
            continue
        class_name = str(class_name).replace(' ', '_') + '_'
        metrics[class_name + "precision"] = row["precision"]
        metrics[class_name + "recall"] = row["recall"]
        metrics[class_name + "iou"] = row["iou"]
        metrics[class_name + "f1"] = row["f1"]
        metrics[class_name + "support"] = int(row["support"])
    return metrics