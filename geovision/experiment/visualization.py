from numpy.typing import NDArray

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.cm import Blues
from matplotlib.axes import Axes
from matplotlib.table import Table
from matplotlib.gridspec import GridSpec 
from matplotlib.widgets import RadioButtons
from matplotlib.animation import FuncAnimation

from geovision.experiment import ExperimentConfig
from geovision.io.local import get_experiments_dir
from geovision.analysis.viz import plot_confusion_matrix

def plot_run_metrics(config: ExperimentConfig):
    fig = plt.figure(figsize = (15, 5), layout = "constrained")
    gs = GridSpec(nrows = 1, ncols = 5, figure = fig)
    ax = fig.add_subplot(gs[:, :4])
    rax = fig.add_subplot(gs[:, 4:])

    with h5py.File(get_experiments_dir(config)/"experiment.h5") as logfile:
        matrices, labels = list(), list()
        for run in sorted(logfile.keys()):
            logs = logfile[run]
            start, step = logs["epoch_begin"][0], logs["log_every_n_epochs"][0]
            if (val_matrices := logs.get("val_confusion_matrix_epoch")) is None:
                continue
            for idx, matrix in enumerate(val_matrices):
                matrices.append(matrix)
                labels.append(f"{run}/epoch={start + step*idx}")

    def draw_matrix(label: str): 
        plot_confusion_matrix(ax, matrices[labels.index(label)])
        ax.set_title(label, fontsize = 10)
        fig.canvas.draw()
        fig.canvas.flush_events()

    buttons = RadioButtons(rax, labels, active = 0)
    buttons.on_clicked(draw_matrix)
    buttons.set_active(0)

#plot_run_metrics(ExperimentConfig.from_config("config.yaml"))

def get_metrics_df(config: ExperimentConfig) -> pd.DataFrame:
    def drop_incomplete_val_metrics(df: pd.DataFrame) -> pd.DataFrame:
        val_epoch, val_start, drop_idxs = False, 0, list() 
        for idx, row in df.iterrows():
            if not val_epoch:
                if pd.notna(row["val/loss_step"]): # val_epoch starts when a val/loss_step is observed 
                    print(f"Cleaner: val started at {idx}")
                    val_epoch, val_start = True, idx # signal that val_epoch has started
                    if idx == len(df) -1: # edge case, if val_epoch begins at the last index
                        drop_idxs.append(idx) # drop the single log
                        val_epoch = False # unneccessary as this is the last entry in the df
                elif pd.notna(row["val/loss_epoch"]): # a lone val_loss/epoch is observed, with no preceesing val_steps  
                    drop_idxs.apend(idx) # drop the val_epoch log

            else:
                if pd.isna(row["val/loss_step"]): # if val_epoch ends
                    print(f"Cleaner: val ended at {idx}")
                    val_epoch = False # signal that val epoch has ended
                    if idx-val_start < 4: # if val_epoch is incomplete
                        drop_idxs.extend(list(range(val_start, idx))) # drop the val_step logs
                        if pd.notna(row["val/loss_epoch"]): # if the incomplete val_epoch ends with epoch loss  
                            drop_idxs.append(idx) # drop the val_epoch log too 
                if idx == len(df) - 1: # if df ends while val_epoch is active 
                    print(f"Cleaner: val ended at {idx}")
                    val_epoch = False # unneccessary as this is the last entry in the df
                    if idx-val_start < 4: # if val_epoch is incomplete
                        drop_idxs.extend(list(range(val_start, idx))) # drop the val_step logs
        drop_idxs = list(set(drop_idxs))
        print(f"Dropping idxs: {drop_idxs}")
        return df.drop(index = list(set(drop_idxs)))

    return (
        pd.read_csv(get_experiments_dir(config) / "metrics.csv")
        .pipe(drop_incomplete_val_metrics)
        .reset_index(drop = True)
        .assign(epoch = lambda df: df["epoch"].ffill().astype("int"))
    )

def get_train_df(metrics_df: pd.DataFrame, metric: str):
    return (
        metrics_df
        .loc[:, ["epoch", "step", "train/loss_step", "train/loss_epoch", f"train/{metric}_epoch"]] # f"train/{config.metric}_step"
        .dropna(subset = ["train/loss_step", "train/loss_epoch"], how = "all")
    )

def get_val_df(metrics_df: pd.DataFrame, metric: str):
    def get_val_steps_per_epoch(df: pd.DataFrame) -> int:
        count = 0
        for _, row in df.iterrows():
            if pd.notna(row["val/loss_step"]):
                count+=1
            else:
                return count

    def assign_val_steps(df: pd.DataFrame) -> pd.DataFrame:
        val_steps_per_epoch = get_val_steps_per_epoch(df)
        step = 0
        for idx, row in df.iterrows():
            if pd.notna(row["val/loss_step"]):
                df.at[idx, "step"] = row["epoch"] * val_steps_per_epoch + step 
                step += 1
            else:
                step = 0
        return df

    return (
        metrics_df
        .loc[:, ["epoch", "step", "val/loss_step", f"val/{metric}_step", "val/loss_epoch", f"val/{metric}_epoch"]]
        .dropna(subset = ["val/loss_step", "val/loss_epoch"], how = "all")
        .pipe(assign_val_steps)
    )

def get_ckpts_df(config: ExperimentConfig) -> pd.DataFrame:
    return (
        pd.DataFrame({"ckpt": [p.name for p in get_experiments_dir(config).rglob("*.ckpt")]})
        .assign(epoch = lambda df: df["ckpt"].apply(lambda x: int(x.split("_")[0].removeprefix("epoch="))))
        .assign(step = lambda df: df["ckpt"].apply(lambda x: int(x.split("_")[1].removeprefix("step=").removesuffix(".ckpt").split('-')[0]) - 1))
    )


def plot_experiment(config: ExperimentConfig, save: bool = True):
    metrics_df = get_metrics_df(config)
    train_df = get_train_df(metrics_df, config.metric)
    val_df = get_val_df(metrics_df, config.metric)
    ckpts_df = get_ckpts_df(config)

    tsdf = train_df[["step", "train/loss_step"]].dropna()
    tedf = train_df[["step", "train/loss_epoch", f"train/{config.metric}_epoch"]].dropna()
    vedf = val_df[["step", "val/loss_epoch", f"val/{config.metric}_epoch"]].dropna()
    vsdf = val_df[["step", "val/loss_step", f"val/{config.metric}_step"]].dropna()
    cedf = ckpts_df[["step", "ckpt"]].dropna()

    style = {
        "train/loss_step": {"color": "deepskyblue", "linewidth": 2},
        "train/loss_epoch": {"color": "dodgerblue", "linewidth": 2},
        "val/loss_step": {"color": "orange", "linewidth": 2},
        "val/loss_epoch": {"color": "darkorange", "linewidth": 2},
        "test/loss_epoch": {"color": "firebrick", "linewidth": 2},

        f"train/{config.metric}_epoch": {"color": "dodgerblue", "linewidth": 2, "linestyle": "dashed"},
        f"val/{config.metric}_epoch": {"color": "darkorange", "linewidth": 2, "linestyle": "dashed"},
        f"test/{config.metric}_epoch": {"color": "firebrick", "linewidth": 2, "linestyle": "dashed"},
    }

    fig, train_ax = plt.subplots(1, 1, figsize = (10, 5), layout = "constrained")
    train_ax.grid(visible = True, axis = "y")
    train_ax.plot(tsdf["step"], tsdf["train/loss_step"], **style["train/loss_step"], label = "train/loss_step")
    train_ax.plot(tedf["step"], tedf["train/loss_epoch"], **style["train/loss_epoch"], label = "train/loss_epoch")
    train_ax.plot(vedf["step"], vedf["val/loss_epoch"], **style["val/loss_epoch"], label = "val/loss_epoch")
    #train_ax.plot(vsdf["step"], vsdf["val/loss_step"], **style["val/loss_step"], label = "val/loss_step")
    for ckpt_step in cedf["step"]:
        train_ax.axvline(ckpt_step, color = "gray", linewidth = 1, linestyle = "dashed")

    _, y_end = train_ax.get_ylim()
    train_ax.set_yticks(np.arange(0, max(1.05, y_end), 0.1))

    epoch_ticks = train_df.groupby("epoch")["step"].max().tolist()
    epoch_ticks = sorted(set(epoch_ticks))
    train_ax.set_xticks(epoch_ticks, labels = [str(x) for x in range(len(epoch_ticks))])
    train_ax.xaxis.set_ticks_position("top")

    #ckpt_ticks = logs_df[["step", "ckpt_path"]].dropna().iloc[:, 0].tolist()
    ckpt_axis = train_ax.secondary_xaxis(location=0)
    ckpt_axis.set_xticks(cedf["step"])

    train_ax.legend(fontsize = 8)#, bbox_to_anchor=(1, 1.01))
    fig.suptitle(f"{config.dataset.name} :: {config.name}" , fontsize = 10)
    plt.show()
    if save:
        fig.savefig(get_experiments_dir(config) / "metrics.png")

# TODO: refactor get_classification_metrics to store metrics in dict and generate df afterwords
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

# def plot_live_experiment(experiment_dir: str | Path):
    # script to spawn a gui and show the live animation
    # get the connection information from ssh config or pass it in via argparse or hardcoded into the script
    # animated plot of the experiment summary with an option to connect and download data over ssh for plotting
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation
    # fig, ax = plt.subplots(1, 1, figsize = (5, 5), layout = "constrained")

    # train_loss_step = []
    # train_step = []

    # val_loss_step = []
    # val_step = []

    # train_plot, = ax.plot(train_step, train_loss_step)
    # ax.set_xlim(0, 200)
    # ax.set_ylim(0, 5)

    # csv_reader = pd.read_csv(experiments_dir / "metrics.csv", chunksize = 10)

    # def update(frame_idx):
        # df = next(csv_reader)
        # tlsdf = df[["step", "train/loss_step"]].dropna()
        # train_loss_step.extend(tlsdf["train/loss_step"].to_list())
        # train_step.extend(tlsdf["step"].to_list())
        # train_plot.set_data(train_step, train_loss_step)
        # ax.autoscale()
        # return [train_plot]

    # ani = FuncAnimation(fig, update, interval = 100, cache_frame_data = False, blit = True)
    # ani
    # pass

# epoch_df = (
    # pd.concat([val_df, train_df, ckpts_df])
    # .loc[:, ["epoch", "step", "train/loss_epoch", "val/loss_epoch", f"train/{config.metric}_epoch", f"val/{config.metric}_epoch", "ckpt"]]
    # .dropna(subset = ["val/loss_epoch", f"val/{config.metric}_epoch", "train/loss_epoch", f"train/{config.metric}_epoch", "ckpt"], how = "all")
    # .groupby(["epoch", "step"], dropna = False).sum(numeric_only = False)
# )
