from typing import Callable, Literal, Optional
from numpy.typing import NDArray

import os
import h5py
import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import multiprocessing as mp

from torch import float32 as f32
from pathlib import Path
from torchvision.io import read_image

# Interface:
#   - extract dimensions and location information for raster images -> extract metadata
#   - perform vectorization of segmentation masks -> raster to vector
#   - perform geo-registeration of bboxes -> raster to vector

def _column(shape: int | tuple[int,...], dtype: Literal["int", "float"] = "int"):
    if dtype == "int":
        return np.empty(shape, dtype = np.int32)
    elif dtype == "float":
        return np.empty(shape, dtype = np.float32)
    else:
        raise ValueError("unexpected :dtype")

def extract_image_dims(df: pd.DataFrame) -> pd.DataFrame:
    """returns a df with height, width and num_channels columns, with the same index as :df. :df expects a column named **image_path**, containing
    valid and absolute paths to images, which can be read by torchvision.io.read_image (PIL). raises OSError if invalid paths found, or KeyError if
    column **image_path** isn't found."""

    stats = ("image_width", "image_height", "image_channels")
    data = {k: _column(len(df)) for k in ("idx",) + stats}
    for i, (idx, row) in enumerate(df.iterrows()):
        data["idx"][i] = idx
        for stat, dim in zip(stats, read_image(row["image_path"]).shape):
            data[stat][i] = dim
    return pd.DataFrame(data).set_index("idx")

def extract_channel_moments(df: pd.DataFrame, num_channels: int = 3) -> pd.DataFrame:
    """returns a df with channel={n}_moment={m} for all (N) channels and the first (M=4) central moments. :df expects a column named **image_path**, containing
    valid and absolute paths to images, which can be read by torchvision.io.read_image (PIL), with the channels as its last (-1) dimension. 
    raises OSError if invalid paths found, or KeyError if column **image_path** isn't found."""
    def load_image(path: Path):
        image = read_image(path).numpy().transpose(1,2,0)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return image

    stats = tuple(f"channel={n}_moment={m}" for n in range(num_channels) for m in range(1,5)) 
    data = {k: _column(len(df)) for k in ("idx", )} | {k: _column(len(df), "float") for k in stats}
    for i, (idx, row) in enumerate(df.iterrows()):
        data["idx"][i] = idx
        image = load_image(row["image_path"]) 
        channel_means = np.mean(image, (0,1))
        for n, moment in zip(range(num_channels), channel_means):
            data[f"channel={n}_moment=1"][i] = moment
        for m in range(2, 5):
            for n, moment in zip(range(num_channels), np.mean(np.pow(image, m), (0,1))):
                data[f"channel={n}_moment={m}"][i] = moment

    return pd.DataFrame(data).set_index("idx")

def extract_location_info(df: pd.DataFrame) -> pd.DataFrame:
    """returns a df with crs, x_gsd, y_gsd, x_off, y_off, with the same index as :df. :df expects a column names **image_path**, containing valid and
    absolute paths to geo-referenced GeoTiff images, which can be read by rasterio (GDAL). raises ..."""

    data = {k: _column(len(df)) for k in ("idx", "crs")} | {k: _column(len(df), "float") for k in ("x_gsd", "y_gsd", "x_off", "y_off")}
    for i, (idx, row) in enumerate(df.iterrows()):
        data["idx"][i] = idx
        with rio.open(row["image_path"]) as image:
            data["crs"][i] = int(image.crs.to_epsg())
            data["x_gsd"][i] = image.transform.a
            data["y_gsd"][i] = image.transform.e
            data["x_off"][i] = image.transform.c
            data["x_off"][i] = image.transform.f

def extract_vector_masks(df: pd.DataFrame) -> pd.DataFrame:
    """returns a df with a geometry (polygon) column, indexed to it's respective source image in :df. :df expects a column named **mask**, containing
    run length encoded binary segmentation masks."""
    pass

def extract_georeferenced_bbox(df: pd.DataFrame) -> pd.DataFrame:
    """returns a georeferenced df with a geometry (polygon) column, indexed to it's respective source image in :df. :df expects columns named **tl_0**,
    **tl_1**, **br_0**, **br_1** representing the top-left and bottom-right corners of the bounding rectangle, along with location information as 
    **crs**, **x_off**, ... usually found in the spatial table.
    """

def extract_metadata(df: pd.DataFrame, fn: Callable, num_processes: Optional[int] = None) -> pd.DataFrame:
    """applies :fn to :df in :num_processes, and returns a concatenated df of results. make sure :fn is defined globally (no closures)"""
    num_processes = num_processes or (os.cpu_count() - 1)
    data = [df.iloc[x] for x in np.array_split(df.index, num_processes)]
    with mp.Pool(processes=num_processes) as pool:
        return pd.concat(pool.map(fn, data))

if __name__ == "__main__":
    cli = argparse.ArgumentParser("data analysis: spatial statistics")
    cli.add_argument('-p', help = "path to .h5 file containing index_df, relative to ~/datasets", required=True, dest = "metadata_path")
    cli.add_argument('-c', help = "statistics to compute", choices=["dimensions", "locations", "moments"], required=True, dest = "compute")
    args = cli.parse_args()

    metadata_path = (Path.home()/"datasets"/ args.metadata_path)
    assert metadata_path.is_file(), f"invalid argument error, :metadata_path does not point to a valid file, got {metadata_path}"

    if args.compute == "dimensions":
        fn = extract_image_dims
    elif args.compute == "locations":
        fn = extract_location_info
    elif args.compute == "moments":
        fn = extract_channel_moments

    index_df = pd.read_hdf(metadata_path, key = "index", mode = 'r')
    stats_df = extract_metadata(index_df, fn)

    if args.compute == "moments":
        stats_df.to_hdf(metadata_path, key = "spectral", mode = "r+")
    else:
        f = pd.HDFStore(metadata_path)
        tables = f.keys()
        f.close()

        if "/spatial" in tables:
            spatial_df = pd.read_hdf(metadata_path, key = "spatial", mode = 'r')
            spatial_df.loc[stats_df.index, stats_df.columns] = stats_df # update spatial_df with data from stats_df
            spatial_df.to_hdf(metadata_path, key = "spatial", mode = 'r+')
        else:
            stats_df.to_hdf(metadata_path, key = "spatial", mode = 'r+')