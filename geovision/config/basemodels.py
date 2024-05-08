from typing import Callable, Optional
import yaml # type:ignore
from pathlib import Path
from pydantic import BaseModel

class DataFrameConfig(BaseModel):
    random_seed: int | None = None
    test_frac: float | None = None
    val_frac: float | None = None
    tile_x: tuple | None = None
    tile_y: tuple | None  = None
    bands: tuple | None = None
    tiling_strategy: str | None = None
    splitting_strategy: str | None = None

    # """return :split_params if valid otherwise raises ValueError or TypeError"""
    # if not isinstance(test_split, int | float):
        # raise TypeError(f":test_split must be numeric, got {type(test_split)}")
    # if not isinstance(val_split, int | float):
        # raise TypeError(f":test_split must be numeric, got {type(test_split)}")
    # if not isinstance(random_seed, int):
        # raise TypeError(f":random seed must be an integer, got {type(random_seed)}")
    # if not (test_split < 1 and test_split >= 0):
        # raise ValueError(f":test_split must belong to [0, 1), received {test_split}")
    # if not (val_split < 1 and val_split >= 0):
        # raise ValueError(f":val_split must belong to [0, 1), received {val_split}")
    # if not (test_split + val_split < 1):
        # raise ValueError(f":test_spilt = {test_split} + :val_split = {val_split} must be < 1")
    # return test_split, val_split, random_seed
  

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    gradient_accumulation: int

class TransformConfig(BaseModel):
    image_transform: Callable | None = None
    target_transform: Callable | None = None
    common_transform: Callable | None = None

class ModelConfig(BaseModel):
    encoder: str
    decoder: str
    weights: str

class LossConfig(BaseModel):
    name: str
    reduction: str | None = None

class OptimizerConfig(BaseModel):
    name: str
    lr: float

class MetricConfig(BaseModel):
    name: str
    mode: str 

class ExperimentConfig(BaseModel):
    experiment_name: str
    dataset_name: str
    dataset_root: Path | None = None
    dataset_df: Path | None = None
    dataframe_params: DataFrameConfig
    dataloader_params: DataLoaderConfig
    transform: TransformConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    metric: MetricConfig

    @classmethod
    def from_yaml(cls, yaml_file: Path, transforms: dict[str, Callable]):
        if not (yaml_file.is_file() and yaml_file.suffix == ".yaml"):
            raise OSError(f":yaml_file is an invalid path to config, got {yaml_file}")
        with open(yaml_file, "r") as config_yaml:
                return cls(**(yaml.safe_load(config_yaml) | transforms))
