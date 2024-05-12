from typing import Optional

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

DATA = Path.home() / "datasets" / "so2sat_pop"
LOC_DIRS = list((DATA/"train").iterdir()) + list((DATA/"test").iterdir())

def _get_file_path(row, subdir) -> Path:
    match subdir.name:
        case "osm":
            file_extn = "osm"
        case "osm_features":
            file_extn = "csv"
        case _:
            file_extn = "tif"
    return subdir/f"Class_{row["label_idx"]}"/f"{row["grd_id"]}_{subdir.name}.{file_extn}"

def _get_loc_df(loc_dir) -> pd.DataFrame:
    return (
        pd.read_csv(loc_dir/f"{loc_dir.name}.csv")
        .set_axis(["grd_id", "label_idx", "pop"], axis = "columns")
        .assign(loc = loc_dir.name.split('_')[-1])
        .assign(split = loc_dir.parent.name)
        .sort_values("label_idx")
        .assign(sen2autumn_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"sen2autumn"), axis = 1))
        .assign(sen2spring_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"sen2spring"), axis = 1))
        .assign(sen2summer_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"sen2summer"), axis = 1))
        .assign(sen2winter_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"sen2winter"), axis = 1))
        .assign(viirs_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"viirs"), axis = 1))
        .assign(lu_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"lu"), axis = 1))
        .assign(lcz_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"lcz"), axis = 1))
        .assign(dem_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"dem"), axis = 1))
        .assign(osm_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"osm"), axis = 1))
        .assign(osm_features_path = lambda df: df.apply(
            lambda x: _get_file_path(x, loc_dir/"osm_features"), axis = 1))
    )

df = pd.concat([_get_loc_df(loc_dir) for loc_dir in LOC_DIRS], axis = 0)
