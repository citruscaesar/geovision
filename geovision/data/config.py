from pathlib import Path
from torch import float32
from pydantic import BaseModel, ConfigDict, field_validator
from geovision.io.local import get_valid_file_err
import torchvision.transforms.v2 as T 

class DatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: str | Path | None = None
    random_seed: int 
    test_sample: int | float | None = None # must be [0,1] if float
    val_sample: int | float | None = None
    tabular_sampling: str | None = None
    tile_x: tuple | None = None
    tile_y: tuple | None  = None
    spatial_sampling: str | None = None
    bands: tuple | None = None
    spectral_sampling: str | None = None
    temporal_sampling: str | None = None
    image_pre: T.Transform | None = None 
    target_pre: T.Transform | None = None 
    train_aug: T.Transform | None = None 
    eval_aug: T.Transform | None = None 

    @field_validator("df")
    @classmethod
    def validate_df(cls, df: str | Path | None = None) -> Path | None:
        return get_valid_file_err(df, valid_extns=(".csv",)) if df is not None else None

class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    gradient_accumulation: int
    persistent_workers: bool
    pin_memory: bool
    prefetch_factor: int