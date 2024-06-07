from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geovision.config.basemodels import ExperimentConfig
from geovision.io.local import get_experiments_dir

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
