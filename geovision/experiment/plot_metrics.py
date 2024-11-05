from typing import Optional
from numpy.typing import NDArray

import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from geovision.io.local import FileSystemIO as fs
from matplotlib.axes import Axes
from matplotlib.cm import tab20b, tab20c, Blues
from matplotlib.table import Table

# plot_metrics.py -> experiment_dir/metrics/...
# 1. Confusion Matrix w/ classwise precision, recall, f1, iou (for each epoch): run={run_idx}_epoch={epoch}_confusion_matrix.png
# 2. Plot all metrics+lr logged (vs step and epoch as available): run={run_idx}_metric={metric_name}.png
# 3. Plot all metrics+lr across all valid runs: metric={metric_name}_runs=[{list_of_runs_dash_separated}].png

# plot_live_metrics.py
# 1. Display window with loss and monitor_metric vs step and epoch (4 graphs)
# 2. Display window with a confusion matrices and radio buttons to choose which one to plot (how to save confm which allows concurrent read/write -> a dir with each in a different file) 

def get_runs(experiment_logs: Path) -> dict[int, tuple[bool, list[str], float]]:
    def is_valid_run(logs) -> bool:
        # metrics should be a set of scalar metrics only, remove confusion_matrix if found
        metrics = get_tracked_metrics(logs)
        if "confusion_matrix" in metrics:
            metrics.remove("confusion_matrix")
        
        # a run is invalid if the logging was terminated unexpectedly -> buffers have extra space -> {split}_{suffix}_end != len(metric_buffer) - 1 
        for split, suffix in ((x,y) for x in ("train", "val", "test") for y in ("step", "epoch")):
            # ideally, last_idx should not exist as it's deleted by trim_run, but if it's found, check the validity condition
            if (last_idx := logs.get(f"{split}_{suffix}_end")) is not None:
                # if a metric buffer is found, check if it's length matches the last index 
                for metric in metrics:
                    if (metric_buffer := logs.get(f"{split}_{metric}_{suffix}")) is not None:
                        if last_idx[0] != len(metric_buffer):    
                            return False
        return True
    
    def get_tracked_metrics(logs) -> list:
        metrics = set()
        for dataset_name in logs:
            parts = dataset_name.split('_')
            if parts[0] in ("train", "val", "test") and parts[-1] in ("step", "epoch"):
                metrics.add(parts[1] if len(parts) == 3 else '_'.join(parts[1:-1]))
        return list(metrics)

    # TODO: fix logger not logging monitor metric for some reason
    def get_monitored_score(logs) -> float:
        if (monitored_metric := logs.get("monitored_metric")) is not None:
            if (monitored_metric_buffer := logs.get(monitored_metric[0])) is not None:
                return monitored_metric_buffer[-1]
        return np.nan

    runs = {"run_idx": list(), "is_valid": list(), "tracked_metrics": list(), "monitored_score": list()} 
    with h5py.File(experiment_logs, mode = "r") as logfile:
        for run in logfile:
            run_logs = logfile[run] 
            runs["run_idx"].append(int(run.removeprefix("run=")))
            runs["is_valid"].append(is_valid_run(run_logs))
            runs["tracked_metrics"].append(get_tracked_metrics(run_logs))
            runs["monitored_score"].append(get_monitored_score(run_logs))
    return runs

def plot_confusion_matrix(logs: Path, run_idx: int, save_to: Path):
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

    def plot(ax: Axes, mat: NDArray, class_names: Optional[tuple] = None, title: Optional[tuple] = None):
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

    with h5py.File(logs, mode = 'r') as logfile:
        run_logs = logfile[f"run={run_idx}"]
        for split in ("train", "val", "test"):
            matrices = run_logs.get(f"{split}_confusion_matrix_epoch")
            if matrices is not None:
                matrices = matrices[:]
                begin, steps_per_epoch  = run_logs["epoch_begin"][0], run_logs[f"num_{split}_steps_per_epoch"][0]
                epochs = np.arange(start = begin + 1, stop = begin + len(matrices) + 1, step = 1)
                for matrix, epoch in zip(matrices, epochs):
                    plot_name = f"run={run_idx}_epoch={epoch}_step={epoch*steps_per_epoch}_confusion_matrix"
                    fig, ax = plt.subplots(1, 1, figsize = (10, 8), layout = "constrained")
                    plot(ax, matrix, class_names=run_logs["class_names"][:], title = plot_name)
                    fig.savefig(save_to/f"{plot_name}.png")

def get_defined_metrics(metric: str, logs):
    metrics = set()
    for split in ("train", "val", "test"):
        for suffix in ("step", "epoch"):
            name = f"{split}_{metric}_{suffix}"
            if logs.get(name) is not None:
                metrics.add(name)
    return metrics

def get_train_metric_steps(run_logs, suffix: str, length: int) -> NDArray:
    if suffix == "step":
        begin, interval = run_logs["step_begin"][0], run_logs["step_interval"][0]
        return np.arange(start = begin + interval, stop = begin + length*interval + 1, step = interval)
    else:
        # interval here is pointless, as epoch logs are logged at every interval
        begin, steps_per_epoch = run_logs["epoch_begin"][0], run_logs["num_train_steps_per_epoch"][0]
        return np.arange(start = begin+1, stop = begin + length+1, step = 1) * steps_per_epoch

def get_eval_metric_steps(prefix: str, run_logs, suffix: str, length: int):
    if suffix == "step":
        epoch_begin = run_logs["epoch_begin"][0]
        step_interval, epoch_interval = run_logs["step_interval"][0], run_logs["epoch_interval"][0]
        train_steps_per_epoch, eval_steps_per_epoch = run_logs["num_train_steps_per_epoch"][0], run_logs[f"num_{prefix}_steps_per_epoch"][0]
        window_len = eval_steps_per_epoch // step_interval
        num_windows = length // window_len
        assert length % window_len == 0, f"val_buffer_len problem {length, window_len}"
        steps = list()
        for idx in range(num_windows):
            begin = (epoch_begin + (epoch_interval * (idx+1) - 1)) * train_steps_per_epoch 
            steps.append(np.arange(start = begin + 1, stop = begin + window_len*step_interval + 1, step = step_interval))
        return np.stack(steps, axis = 0).flatten()
    else: 
        begin, epoch_interval, steps_per_epoch = run_logs["epoch_begin"][0], run_logs["epoch_interval"][0], run_logs["num_train_steps_per_epoch"][0]
        return np.arange(start = begin+1, stop = begin + length*epoch_interval+1, step = epoch_interval) * steps_per_epoch

def get_line_style(split:str, suffix:str):
    style = dict()
    if split == "train":
        style.update({"color": "dodgerblue"})
    elif split == "val":
        style.update({"color": "red"})
    elif split == "test":
        style.update({"color": "green"})
    else:
        style.update({"color": "black"})
    
    if suffix == "step":
        style.update({"linestyle": '-', "linewidth": 1})
    elif suffix == "epoch":
        style.update({"linestyle": 'solid', "linewidth": 2})
    else:
        style.update({"linestyle": ':', "linewidth": 1})
    return style

def plot_scalar_metric(metric: str, logs: Path, run_idx: int, save_to: Path):
    plot_name = f"run={run_idx}_metric={metric}"
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))

    with h5py.File(logs, mode = 'r') as logfile:
        run_logs = logfile[f"run={run_idx}"]
        for metric in get_defined_metrics(metric, run_logs):
            parts = metric.split('_')
            metric_buffer = run_logs[metric][:]
            if parts[0] == "train":
                steps = get_train_metric_steps(run_logs, parts[-1], len(metric_buffer))
                #print(metric, len(metric_buffer), len(steps))
            else:
                steps = get_eval_metric_steps(parts[0], run_logs, parts[-1], len(metric_buffer))
                #print(metric, len(metric_buffer), len(steps))
            ax.plot(steps, run_logs[metric][:], **get_line_style(parts[0], parts[-1]), label = f"{metric}")
        ax.legend()
        ax.set_title(plot_name, fontsize = 10)
    fig.savefig(save_to/f"{plot_name}.png")
    plt.close(fig)

def plot_run_metrics(df: pd.DataFrame, logs: Path, save_to: Path):
    tracked_metrics = set()
    for metrics_list in df["tracked_metrics"]:
        tracked_metrics.update(metrics_list)
    if "confusion_matrix" in tracked_metrics:
        tracked_metrics.remove("confusion_matrix")
    
    # for each metric {accuracy, f1, loss, iou, ...}
    for metric in tracked_metrics:
        plot_name = f"{metric}"
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        # for each metric, iterate over the runs in which metric is tracked
        for run_idx in df[df["tracked_metrics"].apply(lambda x: metric in x)].index:
            with h5py.File(logs, mode='r') as logfile:
                run_logs = logfile[f"run={run_idx}"]
                # for each run, get all the metric buffers ({train, val, test}x{step, epoch}), i.e. max 6
                for _metric in [key for key in run_logs.keys() if metric in key]:
                    parts = _metric.split('_')
                    metric_buffer = run_logs[_metric][:]
                    if parts[0] == "train":
                        steps = get_train_metric_steps(run_logs, parts[-1], len(metric_buffer))
                    else:
                        steps = get_eval_metric_steps(parts[0], run_logs, parts[-1], len(metric_buffer))
                    ax.plot(steps, metric_buffer, **get_line_style(parts[0], parts[-1]), label = f"run={run_idx}_{_metric}")
        ax.legend()
        ax.set_title(plot_name, fontsize = 10)
        fig.savefig(save_to/f"{plot_name}.png")
        plt.close(fig)

if __name__ == "__main__":
    cli = argparse.ArgumentParser("visualize training metrics and such") 
    cli.add_argument("--experiment_dir", "-e", help="path to dir with experiment.h5, as :project_name/:experiment_name relative to ~/experiments/")
    cli.add_argument("--run_idx", "-r", help="(optionally) specify one or a range[) of run_idx, e.g. 4 or 4-7")
    cli.add_argument("--list_runs", "-ls", help="list all the logged runs along with their validity and final metric scores", action = "store_true")
    cli.add_argument("--force_plot_invalid_runs", "-f", help="(optionally) force plotting invalid runs as well", action="store_true")
    args = cli.parse_args()

    experiment_logs = fs.get_valid_file_err(Path.home(), "experiments", args.experiment_dir, "experiment.h5")
    runs_df = pd.DataFrame(get_runs(experiment_logs)).set_index("run_idx")

    if args.list_runs:
        print(runs_df)
        sys.exit()
    if isinstance(args.run_idx, int):
        runs_df = runs_df.iloc[args.run_idx]
    elif isinstance(args.run_idx, str):
        start, end = args.run_idx.split('-')
        runs_df = runs_df.iloc[start: end]
    if not args.force_plot_invalid_runs:
        runs_df = runs_df[runs_df["is_valid"]]

    metrics_dir = fs.get_new_dir(experiment_logs.parent, "metrics")
    for idx, row in runs_df.iterrows():
        run_metrics_dir = fs.get_new_dir(metrics_dir, f"run={idx}")
        for metric in row["tracked_metrics"]:
            if metric == "confusion_matrix":
                plot_confusion_matrix(experiment_logs, idx, run_metrics_dir)
            else:
                plot_scalar_metric(metric, experiment_logs, idx, run_metrics_dir)
    plot_run_metrics(runs_df, experiment_logs, fs.get_new_dir(metrics_dir, f"run={','.join((str(x) for x in runs_df.index))}"))