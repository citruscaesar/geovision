from pathlib import Path
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Transform
from captum.attr import Saliency, visualization

from etl.etl import validate_dir
from training.utils import Loss
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.grid"] = True

from typing import Optional, Any, Literal, Callable
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def metrics_df(confusion_matrix: NDArray, class_names: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    num_classes = confusion_matrix.shape[0]
    num_samples = np.sum(confusion_matrix)

    if isinstance(class_names, tuple):
        assert len(class_names) == num_classes, f"invalid shape, expected len(class_names) = {num_classes}, received = {len(class_names)},"
    else:
        class_names = tuple(str(c) for c in range(num_classes))

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

def metrics_dict(confusion_matrix: NDArray, prefix:str, class_names: Optional[tuple[str, ...]] = None) -> dict[str, float]:
    df = metrics_df(confusion_matrix, class_names)
    metrics = dict()
    for class_name, row in df.iterrows(): # type: ignore
        if str(class_name) == "accuracy":
            metrics[prefix + "accuracy"] = row["precision"]
            continue
        class_name = str(class_name).replace(' ', '_') + '_'
        metrics[prefix + class_name + "precision"] = row["precision"]
        metrics[prefix + class_name + "recall"] = row["recall"]
        metrics[prefix + class_name + "iou"] = row["iou"]
        metrics[prefix + class_name + "f1"] = row["f1"]
        metrics[prefix + class_name + "support"] = int(row["support"])
    return metrics

def plot_metric_table(ax: Axes, df: pd.DataFrame, table_scaling: Optional[tuple[float, float]] = (1., 1.)):
    class_names = df.index[:-3]
    table = ax.table(
        cellText = df.round(3).values, # type: ignore
        rowLabels = tuple(f"{i}: {c}" for i, c in enumerate(class_names)) + ("accuracy", "macro avg", "weighted avg"),
        colLabels = ["precision", "recall", "jaccard", "f1", "support"],
        cellLoc = "center",
        rowLoc = "center",
        loc = "center",
        edges = "horizontal"
    )
    table.scale(*table_scaling)
    table.auto_set_font_size(False)
    table.set_fontsize(10) 
    ax.set_axis_off()

def plot_confusion_matrix(ax: Axes, confusion_matrix: NDArray):
    _num_classes = confusion_matrix.shape[0]
    _font_size = 10 if _num_classes < 30 else 8 

    ax.grid(visible=False)
    ax.imshow(confusion_matrix, cmap = "Blues")
    ax.set_xlabel("Predicted Class", fontsize = _font_size)
    ax.set_xticks(list(range(_num_classes)))
    ax.xaxis.set_label_position("top")

    ax.set_ylabel("True Class", fontsize = _font_size)
    ax.set_yticks(list(range(_num_classes)))
    ax.yaxis.set_label_position("left")

    for r in range(_num_classes):
        for c in range(_num_classes):
            ax.text(y = r, x = c, s = str(confusion_matrix[r, c]), 
                    ha = "center", va = "center", fontsize=_font_size)

def plot_report(confusion_matrix: NDArray, class_names: Optional[tuple[str,...]], step: int = 0, epoch: int = 0) -> Figure:
    assert confusion_matrix.ndim == 2, f"invalid shape, expected confusion_matrix.ndim = 2, received = {confusion_matrix.ndim}"
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1], f"invalid shape, confusion_matrix is not a square matrix"

    num_classes = confusion_matrix.shape[0]
    figsize, table_scaling = get_figsize_table_scaling(num_classes) 

    metric_table = metrics_df(confusion_matrix, class_names)
    fig, (left, right) =  plt.subplots(1, 2, figsize = figsize, width_ratios=(.7, .3))
    fig.suptitle(f"Evaluation Report, step={step}-epoch={epoch}")
    plot_confusion_matrix(left, confusion_matrix);
    plot_metric_table(right, metric_table, table_scaling);
    plt.tight_layout()
    plt.close("all")
    return fig

def get_figsize_table_scaling(num_classes) -> tuple[tuple[int, int], tuple[float, float]]:
    if num_classes <= 3:
        figsize = (12, 5)
        table_scaling = (1, 2)
    elif num_classes <= 12:
        figsize = (14, 7) 
        table_scaling = (0.8, 2)
    elif num_classes <= 15:
        figsize = (16, 8)
        table_scaling = (0.8, 1.9)
    elif num_classes <= 20:
        figsize = (18, 10)
        table_scaling = (0.7, 2)
    else:
        figsize = (21, 15)
        table_scaling = (0.6, 1.7)
    return figsize, table_scaling

def filter_columns(df: pd.DataFrame, monitor_metric: str) -> pd.DataFrame:
    cols = [col for col in df.columns if f"/{monitor_metric}" in col or "loss" in col]
    df["train/loss"] = df["train/loss_step"].add(df["train/loss_epoch"], fill_value = 0)
    df[f"train/{monitor_metric}"] = df[f"train/{monitor_metric}"]
    # TODO: df.drop(["train/loss_step", "train/loss_epoch"], inplace = True)
    return df[["epoch", "step", "train/loss", f"train/{monitor_metric}"]+cols]

def checkpoints_df(logs_dir: Path, monitor_metric: str):
    ckpt_paths = (p for p in Path(logs_dir, "model_ckpts").iterdir() if "last" not in p.stem)
    return (
        pd.read_csv(logs_dir/"metrics.csv")
        # TODO: Use .filter and regex in one line
        .pipe(filter_columns, monitor_metric)
        .set_index(["epoch", "step"])
        .join(
            other = (pd.DataFrame({"ckpt_path": ckpt_paths})
                     .assign(epoch = lambda df: df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[0].removeprefix("epoch="))))
                     .assign(step = lambda df: df["ckpt_path"].apply(lambda x: int(x.stem.split('_')[1].removeprefix("step="))))
                     .set_index(["epoch", "step"])),
            how = "outer")
        .reset_index(drop=False)
    )

def get_x_y(df: pd.DataFrame, x_col: str, y_col: str) -> tuple:
    view = df[[x_col, y_col]].dropna()
    x = view.iloc[:, 0].values
    y = view.iloc[:, 1].values
    return x, y 

def plot_checkpoints(logs_dir: Path, monitor_metric:str, ckpt_metric_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize = (12, 5))
    ax.grid(visible = False, axis = "x")
    ax.plot(*get_x_y(ckpt_metric_df, "step", "train/loss"), label = "train/loss", color="skyblue")
    ax.plot(*get_x_y(ckpt_metric_df, "step", "train/loss_epoch"), label = "train/loss_epoch", color="dodgerblue")
    ax.plot(*get_x_y(ckpt_metric_df, "step", "val/loss"), label = "val/loss", color="darkorange", linewidth = 2)
    ax.plot(*get_x_y(ckpt_metric_df, "step", f"train/{monitor_metric}"), label = f"train/{monitor_metric}", color = "dodgerblue", linewidth = 1, linestyle = "dotted")
    ax.plot(*get_x_y(ckpt_metric_df, "step", f"val/{monitor_metric}"), label = f"val/{monitor_metric}", color="darkorange", linewidth = 1, linestyle = "dotted")

    try:
        ax.scatter(*get_x_y(ckpt_metric_df, "step", "test/loss"), label = "test/loss", color="firebrick", marker = ".")
        ax.plot(*get_x_y(ckpt_metric_df, "step", "test/loss"), label = "test/loss", color="firebrick")
        ax.plot(*get_x_y(ckpt_metric_df, "step", f"test/{monitor_metric}"), label = f"test/{monitor_metric}", color="firebrick", linestyle = "dotted")
        ax.scatter(*get_x_y(ckpt_metric_df, "step", f"test/{monitor_metric}"), label = f"test/{monitor_metric}", color = "firebrick", marker = ".")
    except KeyError:
        pass

    _, y_end = ax.get_ylim()
    ax.set_yticks(np.arange(0, y_end, 0.1))

    _, x_end = ax.get_xlim()
    epoch_ticks = ckpt_metric_df.groupby("epoch")["step"].max().tolist()
    ax.set_xticks(epoch_ticks, labels = [f"{x}" for x in range(len(epoch_ticks))])

    ckpt_ticks = ckpt_metric_df[["step", "ckpt_path"]].dropna().iloc[:, 0].tolist()
    ckpt_axis = ax.secondary_xaxis(location=0)
    ckpt_axis.set_xticks(ckpt_ticks)
    ckpt_axis.tick_params(length = 20)
    for tick in ckpt_ticks:
        ax.axvline(tick, color = "gray", linewidth = 1, linestyle = "dashed")

    #ax.legend(fontsize = 8)
    fig.suptitle("Training Progress")

def samples_df(samples_csv: Path, dataset_csv: Path, split: Literal["correct", "incorrect", "best", "worst", "all"], k: int = 25, **experiment) -> pd.DataFrame:
    assert split in ("correct", "incorrect", "best", "worst", "all"), "invalid split"

    print(f"Loading Dataset From: {dataset_csv}")
    print(f"Loading Samples From: {samples_csv}")
    loss = Loss(experiment.get("loss", "cross_entropy"), experiment.get("loss_params", dict()))
    df = (
        pd.read_csv(dataset_csv)
        .join(other = pd.read_csv(samples_csv, index_col = 0), how = "inner")
        .assign(logits = lambda df: df.apply(lambda x: torch.from_numpy(x[4:].to_numpy(dtype = "float")), axis = 1))
        .assign(pred_idx = lambda df: df["logits"].apply(lambda x: torch.argmax(x, dim = 0).item()))
        .assign(loss = lambda df: df.apply(lambda x: loss(x["logits"], torch.tensor(x["label_idx"])).item(), axis = 1))
        .assign(split = "all")
        .reset_index(drop = True)
    )

    if split == "correct":
        return df[df["label_idx"] == df["pred_idx"]]
    elif split == "incorrect":
        return df[df["label_idx"] != df["pred_idx"]]
    elif split == "best":
        print(f"Returning the best-{k} samples")
        df = df[df["label_idx"] == df["pred_idx"]]
        return df.sort_values("loss", ascending=True).iloc[:min(k, len(df))]
    elif split == "worst":
        print(f"Returning the worst-{k} samples")
        df = df[df["label_idx"] != df["pred_idx"]]
        return df.sort_values("loss", ascending=False).iloc[:min(k, len(df))]
    else:
        return df

def samples_dataloader(dataset_root: Path, dataset_constructor: Callable, samples_df: pd.DataFrame, transform: Optional[Transform]) -> DataLoader:
    return DataLoader(
        dataset = dataset_constructor(dataset_root, split = "all", df = samples_df, common_transform = transform),
        batch_size = 1, 
        shuffle = False,
        pin_memory = True
    )

def plot_checkpoint_attribution(
        model, 
        logs_dir: Path, 
        dataset_root: Path, 
        dataset_constructor: Callable, 
        epoch: int, 
        step: int, 
        split: Literal["correct", "incorrect", "best", "worst", "all"],
        k: int, 
        transforms: Optional[Transform] = None,
        **experiment) -> None:

    # Prepare Data Sources 
    name = f"epoch={epoch}_step={step}"
    dataset_csv = logs_dir / "dataset.csv"
    samples_csv = logs_dir / "eval" / f"{name}_val_samples.csv"
    ckpt_path = Path(logs_dir, "model_ckpts", f"{name}.ckpt")
    assert samples_csv.is_file(), f"dataset: {dataset_csv.name} does not exist"
    assert samples_csv.is_file(), f"samples: {samples_csv.name} does not exist"
    assert ckpt_path.is_file(), f"checkpoint: {ckpt_path.name} does not exist"

    # Prepare Dataloader
    df = samples_df(samples_csv, dataset_csv, split, k = k, **experiment)
    dl = samples_dataloader(dataset_root, dataset_constructor, df, transforms)

    # Prepare Destination Directory
    attrib_dir = validate_dir(logs_dir, "attribution", name, split)

    # Load Model Weights
    state_dict = torch.load(ckpt_path)["state_dict"]
    state_dict = {k.removeprefix("model."): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict)
    model.eval();

    # Call Appropriate Functions
    if experiment["task"] == "classification":
        plot_classification_samples(model, dl, attrib_dir, experiment["class_names"])
    elif experiment["task"] == "segmentation":
        plot_segmentation_samples(model, dl, attrib_dir, experiment["class_names"])

def plot_classification_samples(model: torch.nn.Module, dl: DataLoader, save_dir: Path, class_names: list):
    for image, true_label_idx, df_idx in dl:
        image.requires_grad = True
        true_label_idx = true_label_idx.item()
        df_idx = df_idx.item()

        logits = torch.softmax(model(image).detach().squeeze(), 0)
        pred_label_idx = logits.argmax().item()

        attribution = Saliency(model)
        grads = (attribution.attribute(inputs = image, target = true_label_idx)#, baselines = torch.zeros_like(image))
                            .detach().cpu().squeeze().permute(1,2,0).numpy())
        image = image.detach().cpu().squeeze().permute(1,2,0).numpy()

        fig = plot_classification_sample(image, grads, logits, class_names, true_label_idx, pred_label_idx) 
        fig.savefig(save_dir / f"{df_idx}.png")

def plot_classification_sample(image, grads, logits, class_names, true_label_idx, pred_label_idx) -> Figure:
        fig = plt.figure()
        fig.suptitle(f"True: {class_names[true_label_idx]} ({true_label_idx}) :: Predicted: {class_names[pred_label_idx]} ({pred_label_idx})", fontsize = 10) # type: ignore
        gs = GridSpec(nrows = 2, ncols = 2, height_ratios=[2, 1])
        image_ax = fig.add_subplot(gs[0])
        attribution_ax = fig.add_subplot(gs[1])
        histogram_ax = fig.add_subplot(gs[2:])

        image_ax.imshow(image);
        image_ax.axis("off")

        class_idxs = np.arange(0, len(logits), 1, dtype = "int") 
        histogram_ax.bar(x = class_idxs, height = logits);
        histogram_ax.set_xticks(class_idxs)
        histogram_ax.autoscale(True, "both", True)

        visualization.visualize_image_attr(grads, image, plt_fig_axis=(fig, attribution_ax), method = "blended_heat_map");
        attribution_ax.axis("off")

        plt.tight_layout()
        plt.close("all")
        return fig

def plot_segmentation_samples(model: torch.nn.Module, dl: DataLoader, save_dir: Path, class_names: list):
    for image, mask, df_idx in dl:
        pass
