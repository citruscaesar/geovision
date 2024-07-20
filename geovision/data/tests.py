import torch
from typing import Any
from tqdm import tqdm
from memory_profiler import memory_usage # type: ignore
from matplotlib import pyplot as plt
from geovision.analysis.viz import plot_batch
from geovision.io.local import get_new_dir
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

def test_datamodule(dm: LightningDataModule, limit_batches: int = 0, plot_batches: bool = False) -> None:
    plots_dir = get_new_dir('plots') 
    def test_one_epoch(dl: DataLoader, split: str) -> set[int]:
        max_idx = len(dl) if limit_batches == 0 or limit_batches > len(dl) else limit_batches
        batch_shape: tuple = tuple()
        epoch_idxs: set = set()
        for batch_idx, batch in tqdm(enumerate(dl), total = max_idx, desc = f"testing {split} dataloader"):
            if batch_idx > max_idx:
                break
            if batch_idx == 0:
                batch_shape = tuple(x.shape for x in batch[:2])
            if plot_batches:
                plot_batch(dl.dataset, batch, batch_idx, plots_dir) # type: ignore
            if len(repeated_idxs := epoch_idxs.intersection(batch[2].tolist())) != 0:
                print(f"repeated samples in {split} epoch, indices: {repeated_idxs}")
            epoch_idxs.update(batch[2].tolist())

        print(f"{split} image batch shape: {batch_shape[0]}")
        print(f"{split} label batch shape: {batch_shape[1]}")
        return epoch_idxs
    
    dm.setup("validate")
    df = dm.val_dataset.df # type: ignore 
    all_idxs = set(df.index.to_list())
    dm.setup('fit')
    train_mem_usage, train_idxs = memory_usage((test_one_epoch, (dm.train_dataloader(), "train")), retval=True)
    val_mem_usage, val_idxs = memory_usage((test_one_epoch, (dm.val_dataloader(), "val")), retval=True)

    dm.setup('test')
    test_mem_usage, test_idxs = memory_usage((test_one_epoch, (dm.test_dataloader(), "test")), retval=True)

    plt.plot(train_mem_usage, label = "train")
    plt.plot(val_mem_usage, label = "val")
    plt.plot(test_mem_usage, label = "test")
    plt.legend()
    plt.title("Memory Profile")
    plt.ylabel("Memory Used (MB)")
    plt.xlabel("Time (100ms)")

    observed_idxs = set().union(train_idxs, val_idxs, test_idxs)
    if observed_idxs != all_idxs:
        print("mismatch between dataframe and observed samples")

        unobserved_idxs = all_idxs.difference(observed_idxs)
        if len(unobserved_idxs) != 0:
            print("indices were not observed")
            display(df.filter(items = sorted(unobserved_idxs), axis = 0)) # type: ignore # noqa

        overobserved_idxs = observed_idxs.difference(all_idxs)
        if len(overobserved_idxs) != 0:
            print(f"unknown indices were observerd: {overobserved_idxs}") 

    overlapped_idxs: set = set().intersection(train_idxs, val_idxs, test_idxs)
    if len(overlapped_idxs) != 0:
        print(f"overlapping samples, indices: {overlapped_idxs}")
        display(df.filter(items = sorted(overlapped_idxs), axis = 0)) # type: ignore # noqa

def test_dataset(config: Any, split: str = "all"):
    ds: Dataset = config.get_dataset(split)

    expected_image_shape, expected_label_shape = ds[0][0].shape, ds[0][1].shape
    expected_image_dtype, expected_label_dtype = ds[0][0].dtype, ds[0][1].dtype

    print(f"{ds.name} @ [{ds.root}]") # type: ignore
    print(f"image shape: {expected_image_shape}, dtype: {expected_image_dtype}")
    print(f"label shape: {expected_label_shape}, dtype: {expected_label_dtype}")

    df_idxs: set[int] = set()
    for idx in tqdm(range(len(ds)), desc = "loading dataset"): # type: ignore
        image, label, df_idx = ds[idx]
        df_idxs.add(df_idx)
        if image.shape != expected_image_shape:
            print(f"idx = {df_idx}, inconsistent image shape, got {image.shape}")
        if label.shape != expected_label_shape:
            print(f"idx = {df_idx}, inconsistent label shape, got {label.shape}")
        if image.dtype != expected_image_dtype:
            print(f"idx = {df_idx}, inconsistent image dtype, got {image.dtype}")
        if label.dtype != expected_label_dtype:
            print(f"idx = {df_idx}, inconsistent label dtype, got {label.dtype}")
    
    if len(missing_idxs := set(ds.df.index.to_list()).difference(df_idxs)) != 0: # type: ignore
        print("some samples did not load")
        display(ds.df.filter(items = sorted(missing_idxs), axis = 0))  # type: ignore # noqa