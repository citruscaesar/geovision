from numpy.typing import NDArray
from typing import Any, Optional, Literal, Callable, Sequence
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from functools import cache
from itertools import product
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from pandera.pandas import DataFrameSchema
from torch.utils.data import default_collate
from geovision.io.local import FileSystemIO as fs 
from torchvision.transforms.v2 import Transform, Identity, CutMix, MixUp

#from litdata import StreamingDataLoader, StreamingDataset

logger = logging.getLogger(__name__)

def stratified_tabular_sampler(index_df: pd.DataFrame, test_frac: float, val_frac: float, split_on: str, random_seed: int) -> pd.DataFrame:
    """
    returns df split into train-val-test, by stratified(proportionate) sampling, based on :split_col (class label) such that for each (i^th)
    class, i ∈ {0, ..., num_classes-1} and eval ∈ {val, test}, num_:eval_samples[i] = :eval_frac * num_samples[i]

    Parameters:
        :df -> table to resample
        :random_seed -> to use for deterministic sampling from dataframe
        :test_frac -> proportion of samples per class for testing 
        :val_frac -> proportion of samples per class for validation
        :split_on -> column used to group by, must be present in :df
    
    """ 
    assert split_on in index_df.columns 
    assert isinstance(test_frac, float), f"config error (invalid type), expected :test_frac to be of type float, got {type(test_frac)}"
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert test_frac + val_frac > 0 and test_frac + val_frac < 1, \
        f"config error (invalid value), expected 0 < :test_frac + :val_frac < 1, got :test_frac={test_frac} and :val_frac={val_frac}"
    
    test = (
        index_df
        .groupby(split_on, group_keys=False)
        [index_df.columns]
        .apply(lambda x: x.sample(frac=test_frac, random_state=random_seed, axis=0))
        .assign(split = "test")
    ) 
    val = (
        index_df
        .drop(test.index, axis = 0)
        .groupby(split_on, group_keys=False)
        [index_df.columns]
        .apply(lambda x: x.sample(frac=val_frac/(1-test_frac), random_state=random_seed, axis=0)) # type: ignore
        .assign(split="val")
    )
    train = (
        index_df
        .drop(test.index, axis = 0)
        .drop(val.index, axis = 0)
        .assign(split = "train")
    )

    # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :index_df
    return pd.concat([train, val, test]).sort_index()

def equal_tabular_sampler(index_df: pd.DataFrame, num_val_samples: int, num_test_samples: int, split_on: str, random_seed: int) -> pd.DataFrame:
    assert split_on in index_df.columns 
    assert isinstance(num_val_samples, int) and num_val_samples >= 0
    assert isinstance(num_test_samples, int) and num_test_samples >= 0
    assert num_val_samples + num_test_samples < len(index_df)

    test = (
        index_df
        .groupby(split_on, group_keys=False)
        .apply(lambda x: x.sample(n = num_test_samples, random_state=random_seed, axis=0), include_groups=True)
        .assign(split = "test")
    )

    val = (
        index_df
        .drop(index=test.index)
        .groupby(split_on, group_keys=False)
        .apply(lambda x: x.sample(n = num_val_samples, random_state=random_seed, axis=0), include_groups=True) # type: ignore
        .assign(split="val")
    )

    train = (
        index_df
        .drop(index=test.index)
        .drop(index=val.index)
        .assign(split = "train")
    )

    # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :index_df
    return pd.concat([train, val, test]).sort_index()

def imagenet_tabular_sampler(index_df: pd.DataFrame, val_frac: int, split_on: str, random_seed: int) -> pd.DataFrame:
    """
    returns df where split=test is assigned to samples with 'val' in their paths, and split=val sampled (:val_frac) proportionally from the 
    remaining samples using the column named :split_on as the class label.
    """
    assert split_on in index_df.columns
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert val_frac >= 0.0 and val_frac < 1.0, f"config error (invalid value), expected 0 <= :val_frac < 1, got :val_frac={val_frac}"

    # list all samples inside val/ and assign split = test
    test = index_df.loc[lambda df: df["image_path"].apply(lambda x: "val" in str(x))].assign(split = "test")

    # sample from inside train/ and assign split = val 
    val = (
        index_df.drop(index=test.index)
        .groupby(split_on, group_keys=False)
        .apply(lambda x: x.sample(frac = val_frac, random_state = random_seed, axis = 0), include_groups = True)
        .assign(split = "val")
    )

    # assign split = train to remaining samples
    train = (
        index_df
        .drop(test.index, axis = 0)
        .drop(val.index, axis = 0)
        .assign(split = "train")
    )
    return pd.concat([train, val, test]).sort_index()

def imagefolder_tabular_sampler(
        index_df: pd.DataFrame, 
        random_seed: int, 
        val_split: Optional[int | float] = None,
        split_on: Optional[str] = None 
    ) -> pd.DataFrame:
    """
    returns df by assiging split based on the grand-parent dir, i.e., assuming a ../{split}/{class}/{image} format in :index_df["image_path"].
    if the val/ dir is not found, :val_split samples from the train/ dir are assigned split=val using the :split_on column. 
    if the test/ dir is not found, samples from the val/ dir are assigned split=test, and :val_split samples from the train/ dir are assigned 
    split=val using the :split_on column. raises Assertion error if both test/ and val/ are missing.

    Parameters
    -
    :index_df -> table to resample.
    :random_seed -> for deterministic sampling.
    :val_frac -> if integer, specifies the number of samples from train/ to resample as val/. if float, specifies proportion per 
    class of the samples in train/ to resample as val/ .
    :split_on -> column in :df containing the parent class names, used to group by, defaults to class_dir if not specified

    """
    def get_val(train_df: pd.DataFrame) -> pd.DataFrame:
        val_df = train_df.groupby(split_on, group_keys=False)
        if isinstance(val_split, int):
            val_df = val_df.apply(lambda x: x.sample(n = val_split, random_state=random_seed, axis=0), include_groups=True) # type: ignore
        elif isinstance(val_split, float):
            val_df = val_df.apply(lambda x: x.sample(frac = val_split, random_state=random_seed, axis=0), include_groups=True) # type: ignore
        val_df = val_df.assign(split="val")
        return val_df

    index_df["split"] = index_df["image_path"].apply(lambda x: str(x).split('/')[-3])

    splits = set(index_df["split"].unique())
    if "test" not in splits or "val" not in splits:
        assert split_on in index_df.columns
        assert val_split is not None and isinstance(val_split, (int, float))

        if "test" not in splits:
            assert "train" in splits and "val" in splits
            test = index_df[index_df["split"] == "val"]
        elif "val" not in splits:
            assert "train" in splits and "test" in splits
            test = index_df[index_df["split"] == "test"]
        train = index_df[index_df["split"] == "train"]
        val = get_val(train)
        train = train.drop(index=val.index)
        return pd.concat([train, val, test]).sort_index()
    return index_df

def sliding_window_spatial_sampler(
        index_df: pd.DataFrame,
        spatial_df: pd.DataFrame,
        tile_size: int | Sequence[int],
        tile_stride: int | Sequence[int],
        **kwargs
    ) -> pd.DataFrame:
    """
    returns df with tile_x_min, x_max, y_min and y_max , indicating the pixel coordinates of the top left and bottom right corners of the image 
    tile respectively. these are calculated by sliding a window of :tile_size over the image with :tile_stride. this sampler expects the image 
    bounds to be specified using 'image_width' and 'image_height' columns in the spatial_df.

    Parameters
    -
    :index_df > df to resample\n
    :spatial_df > df with spatial info, specifially image_height and image_width\n 
    :tile_size > size of tile in pixels used to calculate tile bounds. int is converted to tuple[int, int]\n
    :tile_stride > stride of sliding window in pixels used to calculate tile bounds. int is converted to tuple[int, int]\n
    """
    
    @cache
    def get_min(length: int, stride: int) -> NDArray:
        return np.arange(start=0, stop=length, step=stride, dtype=np.uint16)
    
    assert index_df.index.equals(spatial_df.index)
    assert "image_height" in spatial_df.columns
    assert "image_width" in spatial_df.columns

    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(tile_stride, int):
        tile_stride = (tile_stride, tile_stride)

    assert isinstance(tile_size, Sequence) and len(tile_size) == 2
    assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

    top_left = {k:list() for k in ("idx", "y_min", "x_min")}
    for idx, row in spatial_df.iterrows():
        y_mins = get_min(row["image_height"], tile_stride[0])
        x_mins = get_min(row["image_width"], tile_stride[1])
        for y_min, x_min in product(y_mins, x_mins):
            top_left["idx"].append(idx)
            top_left["y_min"].append(y_min)
            top_left["x_min"].append(x_min)
        
    df = pd.DataFrame(top_left).set_index("idx").merge(index_df, how = "left", left_index=True, right_index=True)
    df["y_max"] = df["y_min"] + tile_size[0]
    df["x_max"] = df["x_min"] + tile_size[1]
    df = df[index_df.columns.to_list() + ["y_min", "y_max", "x_min", "x_max"]]
    return df

def band_combination_spectral_sampler(
        index_df: pd.DataFrame,
        spectral_df: pd.DataFrame,
        bands: Sequence[int] 
    ) -> pd.DataFrame:
    """
    returns df with the chosen :bands subset from the image. this fn expects the number of channels in the image to be specified by a column 
    named num_channels in the spectral_df

    Parameters
    -
    :index_df -> table to be resampled
    :spectral_df -> table containing spectral information
    :bands -> bands to be subset from the multispectral image 
    """
    ...

