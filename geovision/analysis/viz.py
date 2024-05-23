from typing import Optional 
from numpy.typing import NDArray

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes 

from geovision.data.dataset import Dataset

def get_standardized_image(image):
    return (image - image.mean()) / image.std()

def get_normalized_image(image):
    return (image - image.min()) / (image.max() - image.min())

def plot_classification_sample(ax: Axes, image, label_idx, label_str, ds_idx, ):
    image = get_normalized_image(image)
    ax.imshow(image.permute(1,2,0))
    ax.set_title(f"label: {label_str} ({label_idx}) :: idx: {ds_idx}", fontsize = 10)
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
    
    fig, axes = plt.subplots(nrows, ncols, figsize = (15, 15), layout = "constrained")
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

def plot_confusion_matrix(ax: Axes, confusion_matrix: NDArray):
    pass

def plot_metrics_table(ax: Axes, metrics_df: pd.DataFrame):
    pass

def plot_experiment(experiment_dir: Path):
    # Summarize the experiment by plotting the train, val, test losses and monitored metric 
    # (optionally) along with any additional metrics curvs, auroc, precision vs recall, etc.
    pass

def plot_inference_sample_for_classification(ax, sample: NDArray, logits: NDArray, class_names: tuple[str]):
    pass

def plot_inference():
    pass

def plot_attribution_sample_for_classification(ax, sample: NDArray, attribution: NDArray, logits: NDArray, class_names: tuple[str]):
    # General plotting function which plots the attribution image with the sample image
    # MIght expand to several functions for different attribution methods and problems/datasets 
    # e.g. plot_attribution_sample_for_segmentation(ax, sample: NDArray)
    # or plot_attribution_sample_for_multispectral_segmentation(ax, sample: NDArray)
    pass

def plot_sampling_dist():
    # Utility to show how a transformation / augmentation shifts the spectral distribution of a dataset 
    # Plots the dataset (population) statistic against the statistic estimated using the sampling distribution of the sample statistic
    # (optionally) with multiple graphs with varying batch sizes to show the effect of the central limit theorem
    pass

