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
from geovision.experiment.utils import plot as cfm_plot

# plot_metrics.py -> experiment_dir/metrics/...
# 1. Confusion Matrix w/ classwise precision, recall, f1, iou (for each epoch): run={run_idx}_epoch={epoch}_confusion_matrix.png
# 2. Plot all metrics+lr logged (vs step and epoch as available): run={run_idx}_metric={metric_name}.png
# 3. Plot all metrics+lr across all valid runs: metric={metric_name}_runs=[{list_of_runs_dash_separated}].png

# plot_live_metrics.py
# 1. Display window with loss and monitor_metric vs step and epoch (4 graphs)
# 2. Display window with a confusion matrices and radio buttons to choose which one to plot (how to save confm which allows concurrent read/write -> a dir with each in a different file) 

def get_runs(experiment_logs: Path) -> dict[int, tuple[bool, list[str], float]]:
    def get_tracked_metrics(logs) -> list:
        metrics = set()
        for dataset_name in logs:
            parts = dataset_name.split('_')
            if parts[0] in ("train", "val", "test") and parts[-1] in ("step", "epoch"):
                metrics.add(parts[1] if len(parts) == 3 else '_'.join(parts[1:-1]))
        return list(metrics)

    def is_valid_run(run_logs) -> bool:
        # a run is invalid if the logging was terminated unexpectedly -> buffers have extra space -> metric_ds.attrs["idx"] is defined -> idx != len(metric_buffer) - 1 
        for metric in run_logs:
            if "confusion_matrix" in metric: 
                continue
            last_idx = run_logs[metric].attrs.get("idx")
            if last_idx is not None and last_idx != len(run_logs[metric]):
                return False
        return True

    # TODO: fix logger not logging monitor metric for some reason
    def get_monitored_score(run_logs) -> float:
        monitored_metric = run_logs.attrs.get("monitored_metric")
        if monitored_metric is not None:
            metrics = sorted(get_defined_metrics(monitored_metric, run_logs), reverse = True)
            return run_logs[metrics[-1]][-1]
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
    with h5py.File(logs, mode = 'r') as logfile:
        run_logs = logfile[f"run={run_idx}"]
        for split in ("train", "val", "test"):
            matrices = run_logs.get(f"{split}_confusion_matrix_epoch")
            if matrices is not None:
                matrices = matrices[:]
                begin, steps_per_epoch  = run_logs.attrs["epoch_begin"], run_logs.attrs[f"num_{split}_steps_per_epoch"]
                epochs = np.arange(start = begin + 1, stop = begin + len(matrices) + 1, step = 1)
                for matrix, epoch in zip(matrices, epochs):
                    plot_name = f"run={run_idx}_epoch={epoch}_step={epoch*steps_per_epoch}_confusion_matrix"
                    fig, ax = plt.subplots(1, 1, figsize = (10, 8), layout = "constrained")
                    cfm_plot(ax, matrix, class_names=run_logs.attrs["class_names"], title = plot_name)
                    fig.savefig(save_to/f"{plot_name}.png")

def get_defined_metrics(metric: str, run_logs):
    metrics = set()
    for split in ("train", "val", "test"):
        for suffix in ("step", "epoch"):
            name = f"{split}_{metric}_{suffix}"
            if run_logs.get(name) is not None:
                metrics.add(name)
    return metrics

def get_train_metric_steps(run_logs, suffix: str, length: int) -> NDArray:
    if suffix == "step":
        begin, interval = run_logs.attrs["step_begin"], run_logs.attrs["step_interval"]
        return np.arange(start = begin + interval, stop = begin + length*interval + 1, step = interval)
    else:
        # interval here is pointless, as epoch logs are logged at every interval
        begin, steps_per_epoch = run_logs.attrs["epoch_begin"], run_logs.attrs["num_train_steps_per_epoch"]
        return np.arange(start = begin+1, stop = begin + length+1, step = 1) * steps_per_epoch

def get_eval_metric_steps(prefix: str, run_logs, suffix: str, length: int):
    if suffix == "step":
        epoch_begin = run_logs.attrs["epoch_begin"]
        step_interval, epoch_interval = run_logs.attrs["step_interval"], run_logs.attrs["epoch_interval"]
        train_steps_per_epoch, eval_steps_per_epoch = run_logs.attrs["num_train_steps_per_epoch"], run_logs.attrs[f"num_{prefix}_steps_per_epoch"]
        window_len = eval_steps_per_epoch // step_interval
        num_windows = length // window_len
        assert length % window_len == 0, f"val_buffer_len problem {length, window_len}"
        steps = list()
        for idx in range(num_windows):
            begin = (epoch_begin + (epoch_interval * (idx+1) - 1)) * train_steps_per_epoch 
            steps.append(np.arange(start = begin + 1, stop = begin + window_len*step_interval + 1, step = step_interval))
        return np.stack(steps, axis = 0).flatten()
    else: 
        begin, epoch_interval, steps_per_epoch = run_logs.attrs["epoch_begin"], run_logs.attrs["epoch_interval"], run_logs.attrs["num_train_steps_per_epoch"]
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