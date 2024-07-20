from typing import Optional, Sequence
from numpy.typing import NDArray

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes 
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import CheckButtons
from matplotlib.colors import ListedColormap, is_color_like
from matplotlib.cm import tab20b, tab20c, Blues
from matplotlib.lines import Line2D
from matplotlib.table import Table

from geovision.data.dataset import Dataset
from geovision.config.config import ExperimentConfig
from geovision.io.local import get_experiments_dir
from geovision.analysis.metrics import get_classification_metrics_df, get_classification_metrics_dict

def get_standardized_image(image):
    return (image - image.mean()) / image.std()

def get_normalized_image(image):
    return (image - image.min()) / (image.max() - image.min())

def plot_classification_sample(ax: Axes, image, label_idx, label_str, ds_idx, ):
    image = get_normalized_image(image)
    ax.imshow(image.permute(1,2,0))
    ax.set_title(f"{label_str}({label_idx})[{ds_idx}]", fontsize = 10)
    ax.axis("off")

def plot_segmentation_sample(ax, image, label, ds_idx):
    ax.imshow(image.permute(1,2,0))
    ax.imshow(label.argmax(0), alpha = 0.5, cmap = "Reds_r")
    ax.set_title(f"idx: {ds_idx}")

def plot_sample_top_k_logits(ax: Axes, logits: NDArray, label: int | NDArray, top_k: int = 0):
    # plot bar chart with logits, highlight the correct label with a different color bar
    pass

def plot_sample(ax: Axes, dataset: Dataset, idx: Optional[int] = None):
    task = dataset.name.split('_')[-1]
    idx = np.random.randint(0, len(dataset)) if idx is None else idx
    image, label, ds_idx = dataset[idx]
    match task:
        case "classification":
            plot_classification_sample(ax, image, label, dataset.class_names[label], ds_idx)
        case "segmentation":
            plot_segmentation_sample(ax, image, label, ds_idx)
 
def plot_batch(dataset: Dataset, batch: tuple, batch_idx: int, save_to: Optional[Path] = None):
    task = dataset.name.split('_')[-1]
    images, labels, ds_idxs = batch
    n = len(images)
    assert n >= 2, f"batch_size must be at least 2, got {n}"
    nrows = int(np.around(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize = (20, 20), layout = "constrained")
    fig.suptitle(f"{dataset.name} :: batch: {batch_idx}", fontsize = 10)
    for idx, ax in enumerate(axes.ravel()):
        if idx < n: 
            image, label, ds_idx = images[idx], labels[idx], ds_idxs[idx]
            match task:
                case "classification":
                    plot_classification_sample(ax, image, label, dataset.class_names[label], ds_idx)
                case "segmentation":
                    plot_segmentation_sample(ax, image, label, ds_idx)
        ax.axis("off")
    if save_to is not None:
        fig.savefig(save_to/f"{dataset.name}_batch={batch_idx}_plot.png")
        plt.clf()
        plt.close()

def get_confusion_matrix_plot(mat: NDArray, class_names: Optional[tuple] = None, title: Optional[tuple] = None) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize = (15, 5), layout = "constrained")
    plot_confusion_matrix(ax, mat, class_names, title)
    return fig

def plot_confusion_matrix(ax: Axes, mat: NDArray, class_names: Optional[tuple] = None, title: Optional[tuple] = None):
    metrics_df = get_classification_metrics_df(get_classification_metrics_dict(mat, class_names))

    if title is not None:
        ax.title(title, fontsize = 10)
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
        metric_table.add_cell(row=r_idx, col=0, width=3*w, height=1/mat.shape[0], text=name, loc="center")
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

def plot_inference_sample_for_classification(ax, sample: NDArray, logits: NDArray, class_names: tuple[str]):
    pass

def plot_inference():
    pass

def plot_attribution_sample_for_classification(ax, sample: NDArray, attribution: NDArray, logits: NDArray, class_names: tuple[str]):
    # General plotting function which plots the attribution image with the sample image
    # Might expand to several functions for different attribution methods and problems/datasets 
    # e.g. plot_attribution_sample_for_segmentation(ax, sample: NDArray)
    # or plot_attribution_sample_for_multispectral_segmentation(ax, sample: NDArray)
    pass

def plot_sampling_dist():
    # Utility to show how a transformation / augmentation shifts the spectral distribution of a dataset 
    # Plots the dataset (population) statistic against the statistic estimated using the sampling distribution of the sample statistic
    # (optionally) with multiple graphs with varying batch sizes to show the effect of the central limit theorem
    pass

def plot_metric_line(ax: Axes, values: NDArray, x_begin: int, x_interval: int, label: str, color: Optional[Sequence] = None):
    if color is not None:
        assert is_color_like(color), f"invalid color: {color}"
    x = x_begin + np.arange(0, x_interval*len(values), x_interval)
    return ax.plot(x, values, label = label, color = color)[0]

def plot_metric(ax: Axes, metric: str, logfile: h5py.File):
    split, freq = metric.split('_')[0], metric.split('_')[-1]
    assert split in ("train", "val", "test"), f"invalid metric name: {metric}"
    assert freq in ("step", "epoch"), f"invalid metric name: {metric}" 

    metric_lines = dict() 
    colormap = ListedColormap(tab20c.colors+tab20b.colors).colors
    for run in sorted(logfile.keys()):
        logs, run_idx = logfile[run], int(run.removeprefix("run_"))
        if freq == "epoch":
            start, stop, step = logs["epoch_begin"][0], logs[f"{split}_epoch_idx"][0], logs["log_every_n_epochs"][0]
            if split == "val" or split == "test":
                color = colormap[4*run_idx] if "loss" in metric else colormap[4*run_idx+1]
            else:
                color = colormap[4*run_idx+2] if "loss" in metric else colormap[4*run_idx+3]
        elif freq == "step":
            start, stop, step = logs["step_begin"][0], logs[f"{split}_step_idx"][0], logs["log_every_n_steps"][0]
            if split == "val" or split == "test":
                color = colormap[4*run_idx] if "loss" in metric else colormap[4*run_idx+1]
            else:
                color = colormap[4*run_idx+2] if "loss" in metric else colormap[4*run_idx+3]
        metric_lines[run] = plot_metric_line(ax, logs[metric][:stop], start, step, run, color)
    return metric_lines

def plot_runs(config: ExperimentConfig):
    def format_axes(ax: Axes):
        ydata = [] 
        for line in ax.get_lines():
            ydata.extend(line.get_ydata())
        if np.median(ydata) < 1.2:
            ax.set_ylim(0, 1.2)
        ax.grid()

    fig = plt.figure(layout = "constrained", figsize = (12, 6))
    gs = GridSpec(nrows = 6, ncols = 12, figure = fig) 
    train_step_ax = fig.add_subplot(gs[:3, :5])
    val_step_ax = fig.add_subplot(gs[3:, :5])
    epoch_ax = fig.add_subplot(gs[:, 5:-1])
    run_ax = fig.add_subplot(gs[:, -1])

    with h5py.File(get_experiments_dir(config)/"experiment.h5") as logfile:
        runs = sorted(logfile.keys())

    lines: list[dict[str, Line2D]] = []
    lines.append(plot_metric(train_step_ax, "train_loss_step", logfile))
    lines.append(plot_metric(train_step_ax, f"train_{config.metric}_step", logfile))
    lines.append(plot_metric(val_step_ax, "val_loss_step", logfile))
    lines.append(plot_metric(val_step_ax, f"val_{config.metric}_step", logfile))
    lines.append(plot_metric(epoch_ax, "train_loss_epoch", logfile))
    lines.append(plot_metric(epoch_ax, f"train_{config.metric}_epoch", logfile))
    lines.append(plot_metric(epoch_ax, "val_loss_epoch", logfile))
    lines.append(plot_metric(epoch_ax, f"val_{config.metric}_epoch", logfile))

    for ax in [train_step_ax, val_step_ax, epoch_ax]:
        format_axes(ax)

    def callback(label):
        for line in lines:
            ln = line[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()

    check = CheckButtons(run_ax, runs, actives = [True]*len(runs))
    check.on_clicked(callback)

    plt.show()