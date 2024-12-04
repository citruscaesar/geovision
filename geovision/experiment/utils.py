from typing import Optional
from numpy.typing import NDArray

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.cm import Blues
from matplotlib.axes import Axes
from matplotlib.table import Table

# TODO: add support for multilabel classification metrics

def plot(ax: Axes, mat: NDArray, class_names: Optional[tuple] = None, title: Optional[tuple] = None):
    def get_classification_metrics_dict(conf_mat: NDArray, class_names: Optional[tuple[str]] = None) -> dict[str, list]:
        num_classes = conf_mat.shape[0]
        num_samples = conf_mat.sum()
        if class_names is None:
            class_names = tuple([f"class_{i:02}" for i in range(num_classes)])
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

    metrics_df = get_classification_metrics_df(get_classification_metrics_dict(mat, class_names))

    ax.imshow(mat, cmap = Blues)
    ax.set_xlabel("Predicted Class", fontsize = 10)
    ax.set_xticks(list(range(mat.shape[0])))
    ax.xaxis.set_label_position("top")

    ax.set_ylabel("True Class", fontsize = 10)
    ax.set_yticks(list(range(mat.shape[0])))
    ax.yaxis.set_label_position("left")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[0]):
            ax.text(y = r, x = c, s = f"{int(mat[r, c]):2d}", ha = "center", va = "center", fontsize=10)

    header_table = Table(ax, loc = "top right")
    for c_idx, c_name in enumerate(("CLS", "P", "R", "F1", "IoU", "sup")):
        w = 1/mat.shape[0]
        header_table.add_cell(row=0, col=c_idx, width=3*w if c_name == "CLS" else w, height=1/mat.shape[0], text=c_name, loc = "center")
    ax.add_table(header_table)

    metric_table = Table(ax, loc = "right")
    for r_idx, (name, row) in enumerate(metrics_df.iloc[:-3].iterrows()):
        name = str(name).removeprefix("b'").removesuffix("'")
        metric_table.add_cell(row=r_idx, col=0, width=3*w, height=1/mat.shape[0], text=str(name), loc="center")
        for c_idx, metric in enumerate(row): 
            if c_idx == 4:
                metric_table.add_cell(row=r_idx, col=c_idx+1, width=w, height=1/mat.shape[0], text=f"{int(metric)}", loc="center")
            else:
                metric_table.add_cell(row=r_idx, col=c_idx+1, width=w, height=1/mat.shape[0], text=f"{metric:.2f}", loc="center", facecolor = Blues(metric))
    ax.add_table(metric_table)

    addl_table = Table(ax, loc = "bottom right")
    for r_idx, (name, row) in enumerate(metrics_df.iloc[-3:].iterrows()):
        addl_table.add_cell(row=r_idx, col=0, width=3*w, height=1/(2*mat.shape[0]), text=name, loc="center")
        for c_idx, metric in enumerate(row): 
            if c_idx == 4:
                addl_table.add_cell(row=r_idx, col=c_idx+1, width=w, height=1/(2*mat.shape[0]), text=f"{int(metric)}", loc="center")
            else: 
                addl_table.add_cell(row=r_idx, col=c_idx+1, width=w, height=1/(2*mat.shape[0]), text=f"{metric:.2f}", loc="center", facecolor = Blues(metric))
    ax.add_table(addl_table)

    if title is not None:
        ax.set_title(title, fontsize = 10)