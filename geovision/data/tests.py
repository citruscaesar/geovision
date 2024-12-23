import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from litdata import StreamingDataLoader
from memory_profiler import memory_usage
from geovision.io.local import FileSystemIO as fs
from geovision.data import Dataset
from geovision.experiment.config import ExperimentConfig
from geovision.analysis.viz import plot_batch

# 1. Plot Samples in Batches
# 2. Test if the shape of the first idx matches the rest, #print shape mismatch / raise error
# 3. Store idxs of splits in sets, check for any overlaps at the end of the epoch
# 4. Test Memory Usage


def test_dataset(dataset: Dataset, batch_size: int, limit_samples: int, save_plots: bool):
    df = dataset.split_df.copy(deep=True).set_index("df_idx")
    image, label, df_idx = dataset[0]

    image_shape = image.shape
    if isinstance(label, int) or isinstance(label, np.integer):
        label_shape = torch.Size([1])
    elif isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
        label_shape = label.shape
    else:
        raise TypeError(f"unexpected label type: expected int or NDArray or Tensor, got {type(label)}")

    print(f"image shape: {image_shape}")
    print(f"label shape: {label_shape}")

    if save_plots:
        plots_dir = fs.get_new_dir("plots", f"{dataset.name}")
    logs_dir = fs.get_new_dir("logs")

    batch_idx = 0
    image_batch = list()
    label_batch = list()
    df_idx_batch = list()
    idxs = {split: set() for split in ("train", "val", "test")}
    errors = {"df_idx": list(), "message": list()}
    for idx in tqdm(range(len(dataset)), total=limit_samples if limit_samples > 0 else len(dataset)):
        if limit_samples > 0 and idx == limit_samples:
            break
        try:
            image, label, df_idx = dataset[idx]
        except Exception as e:  # noqa: E722
            errors["df_idx"].append(df_idx)
            errors["message"].append(e)
            continue
        assert image.shape == image_shape, f"image shape mismatch [{df_idx}]: expected {image_shape} got {image.shape}"
        if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
            assert label.shape == label_shape, f"label shape mismatch [{df_idx}]: expected {label_shape} got {label.shape}"
        else:
            assert isinstance(label, int) or isinstance(label, np.integer), f"type error: expected label to be of type int, got {type(label)}"
        idxs[df.loc[df_idx]["split"]].add(df_idx)

        if save_plots:
            image_batch.append(image)
            label_batch.append(label)
            df_idx_batch.append(df_idx)
            if ((idx + 1) % batch_size == 0) or (idx + 1) == len(dataset):
                plot_batch(dataset, (image_batch, label_batch, df_idx_batch), batch_idx, plots_dir)
                image_batch.clear(), label_batch.clear(), df_idx_batch.clear()
                batch_idx += 1

    if len(errors["df_idx"]) != 0:
        print(f"found errors: {errors}, saving to: {logs_dir}")
        pd.DataFrame(errors).to_csv(logs_dir / (dataset.name + "_errors.csv"), index=False)
    if len(overlap := idxs["train"].intersection(idxs["val"])) != 0:
        print("overlap in train and val splits")
        print(df.filter(items=sorted(overlap), axis=0))
    if len(overlap := idxs["train"].intersection(idxs["test"])) != 0:
        print("overlap in train and test splits")
        print(df.filter(items=sorted(overlap), axis=0))
    if len(overlap := idxs["val"].intersection(idxs["test"])) != 0:
        print("overlap in val and test splits")
        print(df.filter(items=sorted(overlap), axis=0))
    if limit_samples == 0 and len(missing := set(df.index).difference(set().union(idxs["train"], idxs["val"], idxs["test"]))) != 0:
        print("missing samples")
        print(df.filter(items=sorted(missing), axis=0))


def test_dataloader(dataset: Dataset, dataloder_params: dict, limit_batches: int, save_plots: int):
    if "litdata" in dataset.name:
        dataloader = StreamingDataLoader(dataset, shuffle=False, **dataloder_params)
    else:
        dataloader = DataLoader(dataset, shuffle=False, **dataloder_params)
    first = next(iter(dataloader))
    print(f"image shape: {first[0].shape}")
    print(f"label shape: {first[1].shape}")
    for idx, batch in enumerate(tqdm(dataloader, total=limit_batches if limit_batches > 0 else len(dataloader))):
        if limit_batches > 0 and idx == limit_batches:
            break
        elif save_plots:
            plot_batch(dataset, batch, idx, fs.get_new_dir("plots", f"{dataset.name}"))


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser(prog="Data Tester", description="Tests the :dataset class and reports any errors encoutered while loading", epilog="\n")
    cmd_parser.add_argument("--config", default="config.yaml", help="path to config file", dest="config_path")
    cmd_parser.add_argument("--split", choices=["train", "val", "test", "trainvaltest", "all"], default="all", help="which dataset split to test", dest="split")
    cmd_parser.add_argument("--limit", default=0, help="test first n batches / samples", dest="limit")
    cmd_parser.add_argument("--dataloader", action="store_true", help="test dataloader (batches) instead of dataset (samples)", dest="test_dataloader")
    cmd_parser.add_argument("--save_plots", action="store_true", help="save plots of data batches", dest="save_plots")

    args = cmd_parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config_path)
    dataset = config.dataset_constructor(args.split, config.dataset_config)

    if args.test_dataloader:
        print("testing dataloader")
        print(str(config.dataloader_config).replace(" ", "\n"))
        memory_profile = memory_usage((test_dataloader, (dataset, config.dataloader_config.params, int(args.limit), args.save_plots)))
    else:
        print("testing dataset")
        memory_profile = memory_usage((test_dataset, (dataset, config.dataloader_config.batch_size, int(args.limit), args.save_plots)))

    name = f"{dataset.name} memory usage vs time"
    plt.plot(memory_profile)
    plt.title(name)
    plt.ylabel("memory used (MB)")
    plt.xlabel("time (0.1s)")
    plt.savefig(name)
    plt.show()

