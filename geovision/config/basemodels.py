from typing import Any

import yaml # type: ignore
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from torchvision.transforms.v2 import Transform # type: ignore
from geovision.io.local import get_valid_file_err

class DatasetConfig(BaseModel):
    name: str
    root: Path
    df: Path | None = None

    random_seed: int | None = None
    test_sample: int | float | None = None
    val_sample: int | float | None = None
    tabular_sampling: str | None = None

    tile_x: tuple | None = None
    tile_y: tuple | None  = None
    spatial_sampling: str | None = None

    bands: tuple | None = None
    spectral_sampling: str | None = None

    temporal_sampling: str | None = None

class TransformsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_transform: Transform | None = None
    target_transform: Transform | None = None
    common_transform: Transform | None = None

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    gradient_accumulation: int

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
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    transforms: TransformsConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    metric: MetricConfig

    @classmethod
    def from_config_file(cls, config_file: str | Path, transforms: dict[str, Transform | None]):
        config_dict = cls._get_config_dict(config_file)
        transforms_dict = cls._get_transforms_dict(transforms) 
        return cls(**(config_dict | transforms_dict))

    @staticmethod 
    def _get_config_dict(config_file: str | Path) -> dict[str, Any]:
        config_file = get_valid_file_err(config_file, valid_extns=(".yaml",)) 
        with open(config_file, 'r') as config:
            return yaml.safe_load(config)
    
    @staticmethod
    def _get_transforms_dict(transforms: dict[str, Transform | None]) -> dict[str, Any]: 
        return {"transforms": transforms}
