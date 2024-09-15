import pandas as pd
import pandera as pa
from .config import DatasetConfig

import logging
logger = logging.getLogger(__name__)

# TODO: add argument (type and value) validation to all samplers

def tabular_sampler(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    match config.tabular_sampling:
        case None:
            logging.info(":tabular_sampling not specfied, skipping")
            return df
        case "stratified":
            logging.info(":tabular_sampling set to stratified")
            return _get_stratified_split_df(df, config.random_seed, config.test_sample, config.val_sample) # type: ignore
        case "equal":
            logging.info(":tabular_sampling set to equal")
            return _get_equal_split_df(df, config.random_seed, config.test_sample, config.val_sample) # type: ignore
        case "random":
            logging.info(":tabular_sampling set to equal")
            return _get_random_split_df(df, config.random_seed) # type: ignore
        case "imagefolder":
            logging.info(":tabular sampling set to imagefolder")
            return _get_imagefolder_split_df(df)
        case "imagefolder_notest":
            logging.info(":tabular sampling set to imagefolder_notest")
            return _get_imagefolder_notest_split_df(df, config.random_seed, config.val_sample)
        case _:
            raise ValueError(f"invalid :tabular_sampling, must be one of stratified, equal, random, imagenet or None, got {config.tabular_sampling}")

def spatial_sampler(df: pd.DataFrame, config: DatasetConfig):
    match config.spatial_sampling:
        case None:
            logging.info(":spatial_sampling not specfied, skipping")
            return df
        case "image":
            return df
        case "geographic":
            return df

def spectral_sampler(df: pd.DataFrame, config: DatasetConfig):
    match config.spectral_sampling:
        case None: 
            logging.info(":spectral_sampling not specfied, skipping")
            return df
        case "sen2rgb":
            pass
        case "sen2ms":
            pass

def temporal_sampler(df: pd.DataFrame, config: DatasetConfig):
    match config.temporal_sampling:
        case None: 
            logging.info(":temporal_sampling not specified, skipping")
            return df
        case "summer":
            pass


def _get_stratified_split_df(df: pd.DataFrame, random_seed: int, test_frac: float, val_frac: float) -> pd.DataFrame:
    """returns df split into train-val-test, by stratified(proportionate) sampling, based on
    :split_col and :split_config 
    s.t. num_eval_samples[i] = eval_split * num_samples[i], i -> {0, ..., num_classes-1}.

    Parameters
    - 
    :df -> dataframe to split, must contain string typed column named "split_on"
    :random_seed -> to use for deterministic sampling from dataframe
    :test_frac -> proportion of samples per class for testing 
    :val_frac -> proportion of samples per class for validation
    
    Returns
    -
    pd.DataFrame 

    Raises
    -
    SchemaError
    """ 
    df = pa.DataFrameSchema({"split_on": pa.Column(str)})(df)
    assert isinstance(test_frac, float), f"config error (invalid type), expected :test_frac to be of type float, got {type(test_frac)}"
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert test_frac + val_frac > 0 and test_frac + val_frac < 1, f"config error (invalid value), expected 0 < :test_frac + :val_frac < 1, got :test_frac={test_frac} and :val_frac={val_frac}"

    test = (df
            .groupby("split_on", group_keys=False)
            .apply(func = lambda x: x.sample(
                    frac = test_frac, random_state = random_seed, axis = 0),
                    include_groups = False
                )
            .assign(split = "test")) 
            
    val = (df
            .drop(test.index, axis = 0)
            .groupby("label_str", group_keys=False)
            .apply(func = lambda x: x.sample( # type: ignore
                    frac = val_frac / (1-test_frac), random_state = random_seed, axis = 0),
                    include_groups = False
                )
            .assign(split = "val"))

    train = (df
            .drop(test.index, axis = 0)
            .drop(val.index, axis = 0)
            .drop(columns = "split_on")
            .assign(split = "train"))

    # TODO: DO NOT use .reset_index(), since data is stored in the exact order as :df
    return pd.concat([train, val, test]).sort_index()

def _get_random_split_df(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    return pd.DataFrame()

def _get_equal_split_df(df: pd.DataFrame, random_seed: int, test_sample: int, val_sample: int) -> pd.DataFrame:
    return pd.DataFrame()

def _get_imagefolder_split_df(df: pd.DataFrame) -> pd.DataFrame:
    """returns df by assiging splits based on top level parent dir names, assuming {split}/{class_dir}/{sample_image}
    format in df["image_path"]"""
    return df.assign(split = lambda df: df["image_path"].apply(lambda x: x.split('/')[0]))

def _get_imagefolder_notest_split_df(df: pd.DataFrame, random_seed:int, val_frac: float) -> pd.DataFrame:
    """returns df by assigning test split to samples in the val/ directory, and val samples are assigned based on :val_frac
    in a stratified manner from the train/ directory; rest are assigned train"""

    df = pa.DataFrameSchema({"split_on": pa.Column(str)})(df)
    assert isinstance(val_frac, float), f"config error (invalid type), expected :val_frac to be of type float, got {type(val_frac)}"
    assert val_frac > 0 and val_frac < 1.0, f"config error (invalid value), expected 0 < :val_frac < 1, got :val_frac={val_frac}"

    test_filter = df["image_path"].apply(lambda x: "val" in str(x))
    if len(test_filter) == 0:
        raise KeyError("couldn't find samples inside 'val/' dir in the df")
    test = (
        df[test_filter]
        .assign(split = "test")
    )
    val = (
        df.drop(test.index, axis = 0)
        .groupby("split_on", group_keys=False)
        .apply(lambda x: x.sample(frac = val_frac, random_state = random_seed, axis = 0), include_groups = False)
        .assign(split = "val")
    )
    train = (
        df
        .drop(test.index, axis = 0)
        .drop(val.index, axis = 0)
        .assign(split = "train")
    )
    return pd.concat([train, val, test]).drop(columns = "split_on").sort_index()