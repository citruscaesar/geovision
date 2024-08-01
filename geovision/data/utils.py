from typing import Optional

import pandas as pd
import pandera as pa
from pathlib import Path

from .dataset import Dataset
from .config import DatasetConfig
from .samplers import tabular_sampler, spatial_sampler, spectral_sampler, temporal_sampler 

import logging
logger = logging.getLogger(__name__)

def get_valid_split(split: str) -> str:
    """returns split if valid, otherwise raises ValueError"""
    if split not in Dataset.valid_splits: 
        raise ValueError(f":split must be one of {Dataset.valid_splits}, got {split}")
    return split

def get_df(config: DatasetConfig, schema: pa.DataFrameSchema, default_df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(config.df, Path):
            logger.info(f"loading df csv from: {config.df}")
            df = pd.read_csv(config.df)
        elif config.df is None:
            logger.info("using default df")
            df = (
                default_df
                .pipe(tabular_sampler, config)
                .pipe(spatial_sampler, config)
                .pipe(spectral_sampler, config)
                .pipe(temporal_sampler, config)
            )
        else:
            raise TypeError(f":config.df must be a Path or None, got{type(df)}")
        return schema(df) 

def get_split_df(df: pd.DataFrame, schema: pa.DataFrameSchema, split: str, root: Optional[Path] = None):
    split_df = (
        df
        .assign(df_idx = lambda df: df.index)
        .pipe(_get_split_df, split)
    )
    if root is not None:
        split_df = split_df.pipe(_get_root_prefixed_to_df_paths, root)
    return schema(split_df)
            
def _get_split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """returns a subset of rows where :split matches :df.split, raises SchemaError if 'split' column is missing from df"""
    df = pa.DataFrameSchema({"split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))})(df)
    if split == "all":
        return df
    elif split == "trainval":
        return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
    return (df[df.split == split].reset_index(drop=True))

def _get_root_prefixed_to_df_paths(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """prefixes :root to the image_path and mask_path columns in :df"""
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].apply(lambda x: str(Path(root, x)))
    if "mask_path" in df.columns:
        df["mask_path"] = df["mask_path"].apply(lambda x: str(Path(root, x)))
    return df