from typing import Any
import torch
from .dataset import Dataset
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from geovision.config.basemodels import ExperimentConfig
from geovision.analysis.viz import plot_batch
from geovision.io.local import get_new_dir
from tqdm import tqdm

class ImageDatasetDataModule(LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
    
    def setup(self, stage):
        _valid_stages = ("fit", "validate", "test", "predict")
        if stage not in _valid_stages:
            raise ValueError(f":stage must be one of {_valid_stages}, got {stage}")
        if stage in ("fit", "validate"):
            self.val_dataset = self.config.dataset(split = "val", **self._get_dataset_kwargs()) 
            if stage == "fit":
                self.train_dataset = self.config.dataset(split = "train", **self._get_dataset_kwargs())
        if stage == "test":
            self.test_dataset = self.config.dataset(split = "test", **self._get_dataset_kwargs())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.train_dataset,  
            shuffle = True,
            **self._get_dataloader_kwargs()
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = self.val_dataset,
            shuffle = False,
            **self._get_dataloader_kwargs()
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset = self.test_dataset,
            shuffle = False,
            **self._get_dataloader_kwargs()
        )

    def _get_dataset_kwargs(self) -> dict[str, Any]:
        return {
            "root": self.config.dataset_root,
            "df": self.config.dataset_df,
            "config": self.config.dataset_params,
            "transforms": self.config.transforms
        }

    def _get_dataloader_kwargs(self) -> dict:
        kwargs = self.config.dataloader_params.model_dump().copy()
        kwargs["batch_size"] = kwargs["batch_size"] // kwargs["gradient_accumulation"]
        del kwargs["gradient_accumulation"]
        return kwargs 

    # Test output shape, dtype is consistent for all batches per epoch 
    # if batch_idx == 0:
        # epoch_image_shape = images.shape
        # epoch_label_shape = labels.shape
    # elif images.shape != epoch_image_shape:
        # print(f"inconsistent image batch ({batch_idx}) shape, expected {epoch_image_shape}, got {images.shape}")
    # elif labels.shape != epoch_label_shape:
        # print(f"inconsistent label batch ({batch_idx}) shape, expected {epoch_image_shape}, got {images.shape}")
    # epoch_image_shape = torch.Size() 
    # epoch_label_shape = torch.Size() 
    # print(f"{split} image batch shape: {epoch_image_shape}")
    # print(f"{split} label batch shape: {epoch_label_shape}")

def test_datamodule(dm: LightningDataModule, limit_batches: int = 0, plot_batches: bool = False) -> None:
    from memory_profiler import memory_usage
    from matplotlib import pyplot as plt
    # Test for no memory leaks
    # Display memory usage
    plots_dir = get_new_dir('plots') 

    def test_one_epoch(dl: DataLoader, split: str) -> set[int]:
        max_idx = len(dl) if limit_batches == 0 or limit_batches > len(dl) else limit_batches
        batch_shape = tuple()
        epoch_idxs = set()
        for batch_idx, batch in tqdm(enumerate(dl), total = len(dl), desc = f"testing {split} dataloader"):
            if batch_idx > max_idx:
                break
            if batch_idx == 0:
                batch_shape = tuple(x.shape for x in batch[:2])
            if plot_batches:
                plot_batch(dl.dataset, batch, batch_idx, plots_dir)
            if len(repeated_idxs := epoch_idxs.intersection(batch[2].tolist())) != 0:
                print(f"repeated samples in {split} epoch, indices: {repeated_idxs}")
            epoch_idxs.update(batch[2].tolist())

        print(f"{split} image batch shape: {batch_shape[0]}")
        print(f"{split} label batch shape: {batch_shape[1]}")
        return epoch_idxs
    
    dm.setup("validate")
    df = dm.val_dataset.df
    all_idxs = set(df.index.to_list())
    dm.setup('fit')
    train_mem_usage, train_idxs = memory_usage((test_one_epoch, (dm.train_dataloader(), "train")), retval=True)
    val_mem_usage, val_idxs = memory_usage((test_one_epoch, (dm.val_dataloader(), "val")), retval=True)

    dm.setup('test')
    #test_idxs = test_one_epoch(dm.test_dataloader(), "test")
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
            display(df.filter(items = sorted(unobserved_idxs), axis = 0))

        overobserved_idxs = observed_idxs.difference(all_idxs)
        if len(overobserved_idxs) != 0:
            print(f"unknown indices were observerd: {overobserved_idxs}") 

    overlapped_idxs = set().intersection(train_idxs, val_idxs, test_idxs)
    if len(overlapped_idxs) != 0:
        print(f"overlapping samples, indices: {overlapped_idxs}")
        display(df.filter(items = sorted(overlapped_idxs), axis = 0))