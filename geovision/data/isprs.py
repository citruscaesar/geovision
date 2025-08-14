from typing import Literal, Optional, Callable, Sequence
from numpy.typing import NDArray

import h5py
import py7zr
import torch
import shutil
import zipfile
import requests
import subprocess
import numpy as np
import pandas as pd
import pandera as pa
import rasterio as rio
import geopandas as gpd
import imageio.v3 as iio
import torchvision.transforms.v2 as T

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from affine import Affine
from shapely import Polygon
from rasterio.crs import CRS
from rasterio import MemoryFile
from rasterio.features import sieve, shapes
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url
from skimage.measure import find_contours, approximate_polygon

from geovision.data import Dataset, DatasetConfig
from geovision.data.transforms import SegmentationCompose

class ISPRS:
    local = Path.home() / "datasets" / "isprs"

    @classmethod
    def extract(cls, subset: Literal["vaihingen", "potsdam", "toronto"]):
        urls = {
            "potsdam": "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/",
            "toronto": "https://seafile.projekt.uni-hannover.de/f/fc62f9c20a8c4a34aea1/",
            "vaihingen": "https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/",
        }
        #urls = {
            #"potsdam": "https://seafile.projekt.uni-hannover.de/seafhttp/files/652ec9e8-3b26-4f70-bcce-68867b2cf1aa/Potsdam.zip"
        #}
        assert subset in ("vaihingen", "potsdam", "toronto")

        with requests.Session() as session:
            payload = {"csrfmiddlewaretoken": session.get(urls[subset]).cookies.get("sfcsrftoken"), "password": "CjwcipT4-P8g"}
            #response = session.get(urls[subset]+"?dl=1", data = payload)

            response = session.get(urls[subset], data = payload)
            response.raise_for_status()
            local = fs.get_new_dir(cls.local, "staging") / f"{subset}.zip"
            content_len = int(response.headers.get('content-length', 0))
            print(response.headers)
            print(content_len)
            with open(local, "wb") as f, tqdm(desc=f"downloading {local}", total=content_len, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    @classmethod
    def download(cls): ...

    @classmethod
    def load(
        cls,
        table: Literal["index", "spatial"],
        src: Literal["archive", "imagefolder", "hdf5"],
        subset: Literal["vaihingen", "potsdam", "toronto"]
    ) -> pd.DataFrame:
        """
        """
        assert table in ("index", "spatial")
        assert src in ("archive", "imagefolder", "hdf5")
        assert subset in ("vaihingen", "potsdam", "toronto")

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/f"{subset}.h5", key = table, mode = 'r')

        if subset == "vaihingen":
            if src == "archive":
                metadata_path = cls.local/"staging"/f"{subset}_metadata.h5"
                try: 
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    assert table in ("index", "spatial")
                    archive_path = cls.local/"staging"/"vaihingen.zip"
                    with zipfile.ZipFile(archive_path) as zf_outer:
                        metadata = {k:list() for k in ("image_path", "image_height", "image_width")}
                        with zipfile.ZipFile(BytesIO(zf_outer.read('Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip')), 'r') as zf_inner:
                            for image_path in [x for x in zf_inner.namelist() if x.startswith("top/") and x.endswith(".tif")]:
                                with rio.open(MemoryFile(zf_inner.read(image_path)), mode='r') as raster:
                                    metadata["image_path"].append(image_path.removeprefix("top/"))
                                    metadata["image_height"].append(raster.height)
                                    metadata["image_width"].append(raster.width)
                    
                df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop=True)
                index_df = df[["image_path"]].assign(location = "vaihingen")
                spatial_df = df[["image_height", "image_width"]]

                index_df.to_hdf(metadata_path, key = "index", mode = "a")
                spatial_df.to_hdf(metadata_path, key = "spatial", mode = "a")
                return index_df if table == "index" else spatial_df

            elif src == "imagefolder":
                metadata_path = fs.get_valid_dir_err(cls.local, "imagefolder", subset) / "metadata.h5"
                try:
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    imagefolder_path = metadata_path.parent
                    metadata = {k:list() for k in ("image_path", "image_height", "image_width")}
                    for image_path in (imagefolder_path/"images").glob("*.tif"):
                        with rio.open(image_path, mode = 'r') as raster:
                            metadata["image_path"].append('/'.join(image_path.split('/')[-2:]))
                            metadata["image_height"].append(raster.height)
                            metadata["image_width"].append(raster.width)
                    
                df = (
                    pd.DataFrame(metadata)
                    .assign(mask_path = lambda df: df["image_path"].apply(lambda x: x.replace("images", "masks")))
                    .assign(location = "vaihingen")
                    .sort_values("image_path")
                )
                index_df = df[["image_path", "mask_path", "location"]]
                spatial_df = df[["image_height", "image_width"]]

                index_df.to_hdf(metadata_path, key = "index", mode = "a")
                spatial_df.to_hdf(metadata_path, key = "spatial", mode = "a")
                return index_df if table == "index" else spatial_df
            
        elif subset == "potsdam":
            if src == "archive":
                metadata_path = cls.local/"staging"/f"{subset}_metadata.h5"
                try: 
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    assert table in ("index", "spatial")
                    archive_path = cls.local/"staging"/f"{subset}.zip"

                    metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}
                    with zipfile.ZipFile(archive_path) as zf_outer:
                        with zipfile.ZipFile(BytesIO(zf_outer.read("Potsdam/4_Ortho_RGB.zip"))) as zf_inner:
                            for image_path in tqdm([x for x in zf_inner.namelist() if x.endswith(".tif")]):
                                metadata["image_path"].append(image_path.removeprefix("4_Ortho_RGB/"))
                                with rio.open(MemoryFile(zf_inner.read(image_path))) as raster:
                                    metadata["image_width"].append(raster.width)
                                    metadata["image_height"].append(raster.height)
                                    metadata["x_off"].append(raster.transform.xoff)
                                    metadata["y_off"].append(raster.transform.yoff)
                                    metadata["x_res"].append(raster.transform.a)
                                    metadata["y_res"].append(raster.transform.e)
                                    metadata["crs"].append(raster.crs.to_epsg())

                    df = (
                        pd.DataFrame(metadata)
                        .assign(mask_path = lambda df: df["image_path"].apply(lambda x: x.replace("RGB", "label")))
                        .assign(location = "potsdam")
                        .sort_values("image_path")
                    )
                    index_df = df[["image_path", "mask_path", "location"]]
                    spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                    index_df.to_hdf(metadata_path, key = "index", mode = "a")
                    spatial_df.to_hdf(metadata_path, key = "spatial", mode = "a")
                    return index_df if table == "index" else spatial_df

            elif src == "imagefolder":
                metadata_path = fs.get_valid_dir_err(cls.local, "imagefolder", subset) / "metadata.h5"
                try:
                    return pd.read_hdf(metadata_path, key = table, mode = 'r')
                except OSError:
                    imagefolder_path = metadata_path.parent
                    metadata = {k:list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}
                    for image_path in (imagefolder_path/"images").glob("*.tif"):
                        metadata["image_path"].append('/'.join(str(image_path).split('/')[-2:]))
                        with rio.open(image_path, mode = 'r') as raster:
                            metadata["image_height"].append(raster.height)
                            metadata["image_width"].append(raster.width)
                            metadata["x_off"].append(raster.transform.xoff)
                            metadata["y_off"].append(raster.transform.yoff)
                            metadata["x_res"].append(raster.transform.a)
                            metadata["y_res"].append(raster.transform.e)
                            metadata["crs"].append(raster.crs.to_epsg())
                df = (
                    pd.DataFrame(metadata)
                    .assign(mask_path = lambda df: df["image_path"].apply(lambda x: x.replace("images", "masks").replace("RGB", "label")))
                    .assign(location = "vaihingen")
                    .sort_values("image_path")
                )
                index_df = df[["image_path", "mask_path", "location"]]
                spatial_df = df[["image_height", "image_width"]]
                index_df.to_hdf(metadata_path, key = "index", mode = "a")
                spatial_df.to_hdf(metadata_path, key = "spatial", mode = "a")
                return index_df if table == "index" else spatial_df

        elif subset == "toronto": 
            raise NotImplementedError
            
    @classmethod
    def transform(
            cls,
            to: Literal["imagefolder", "hdf5"],
            subset: Literal["vaihingen", "potsdam", "toronto"],
            tile_size: Optional[int | tuple[int, int]] = None,
            tile_stride: Optional[int | tuple[int, int]] = None
        ):
        assert to in ("imagefolder", "hdf5")
        assert subset in ("vaihingen", "potsdam", "toronto")

        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            assert isinstance(tile_size, Sequence) and len(tile_size) == 2

        elif tile_stride is not None:
            if isinstance(tile_stride, int):
                tile_stride = (tile_stride, tile_stride)
            assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        if subset == "vaihingen":
            def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame):
                return (
                    DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                    .assign(image_height = tile_size[0])
                    .assign(image_width = tile_size[1])
                    .reset_index(drop = True)
                )

            archive_path = cls.local / "staging" / f"{subset}.zip" 
            index_df = cls.load('index', 'archive', subset) 
            spatial_df = cls.load('spatial', 'archive', subset) 
            image_archive = 'Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip'
            mask_archive = 'Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip'

            if to == "imagefolder":
                imagefolder_path = cls.local / "imagefolder" / subset
                image_dir_path = fs.get_new_dir(imagefolder_path, "images")
                mask_dir_path = fs.get_new_dir(imagefolder_path, "masks")

                with zipfile.ZipFile(archive_path, mode = 'r') as zf_outer:
                    if tile_size is not None and tile_stride is not None:
                        tiled_df = _tile(index_df, spatial_df)
                        with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                            for image_path in tqdm(tiled_df["image_path"].unique()):
                                image = iio.imread(zf_inner.read(f"top/{image_path}"), extension=".tif")
                                for _, row in tiled_df[tiled_df["image_path"] == image_path].iterrows():
                                    iio.imwrite(image_dir_path/f"{image_path}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif", cls.read_tile(image, row))

                        with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                            for mask_path in tqdm(tiled_df["image_path"].unique()):
                                mask = iio.imread(zf_inner.read(mask_path), extension=".tif")
                                for _, row in tiled_df[tiled_df["image_path"] == mask_path].iterrows():
                                    iio.imwrite(mask_dir_path/f"{mask_path}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif", cls.read_tile(mask, row))
                    else:
                        with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                            for image_path in tqdm((x for x in zf_inner.namelist() if x.startswith("top/") and x.endswith(".tif")), total = 33):
                                zf_inner.extract(image_path, image_dir_path/image_path.split('/')[-1])

                        with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                            for mask_path in tqdm(zf_inner.namelist(), total = 33):
                                zf_inner.extract(mask_path, mask_dir_path/mask_path)

            elif to == "hdf5":
                hdf5_path = cls.local / "hdf5" / f"{subset}.h5"

                with zipfile.ZipFile(archive_path, mode = 'r') as zf_outer:
                    with h5py.File(hdf5_path, mode = 'w') as f:
                        if tile_size is not None and tile_stride is not None:
                            tiled_df = _tile(index_df, spatial_df)
                            images = f.create_dataset('images', (len(tiled_df), *tile_size, 3))
                            masks = f.create_dataset('masks', len(tiled_df), dtype = h5py.special_dtype(vlen=np.uint8))
                            vegetation_masks = f.create_dataset('vegetation_masks', len(tiled_df), dtype = h5py.special_dtype(vlen=np.uint8))

                            with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                                for image_path in tqdm(tiled_df["image_path"].unique()):
                                    image = iio.imread(zf_inner.read(f"top/{image_path}"), extension=".tif")
                                    for idx, row in tiled_df[tiled_df["image_path"] == image_path].iterrows():
                                        images[idx] = cls.read_tile(image, row)

                            with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                                for mask_path in tqdm(tiled_df["image_path"].unique()):
                                    mask = iio.imread(zf_inner.read(mask_path), extension=".tif")
                                    for _, row in tiled_df[tiled_df["image_path"] == mask_path].iterrows():
                                        mask_tile = cls.read_tile(mask, row)
                                        masks[idx] = cls.encode_binary_mask(mask_tile[:, : ,1].squeeze())
                                        vegetation_masks[idx] = cls.encode_binary_mask(mask_tile[:, :, 2].squeeze())
    
                            tiled_df[index_df.columns].to_hdf(hdf5_path, key = "index", mode = "r+")
                            tiled_df[spatial_df.columns].to_hdf(hdf5_path, key = "spatial", mode = "r+")

                        else:
                            images = f.create_dataset('images', len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))
                            masks = f.create_dataset('masks', len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))
                            vegetation_masks = f.create_dataset('vegetation_masks', len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))

                            with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                                for idx, row in tqdm(index_df.iterrows()):
                                    images[idx] = iio.imread(zf_inner.read(f"top/{image_path}"), extension='.tif').flatten()

                            with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                                for idx, row in tqdm(index_df.iterrows()):
                                    mask = iio.imread(zf_inner.read(image_path), extension='.tif')
                                    masks[idx] = cls.encode_binary_mask(mask[:, :, 1]).squeeze() 
                                    vegetation_masks[idx] = cls.encode_binary_mask(mask[:, :, 2].squeeze())
                            index_df.to_hdf(hdf5_path, key = "index", mode = "r+")
                            spatial_df.to_hdf(hdf5_path, key = "spatial", mode = "r+")

        elif subset == "potsdam":
            archive_path = cls.local / "staging" / f"{subset}.zip" 
            index_df = cls.load('index', 'archive', subset) 
            spatial_df = cls.load('spatial', 'archive', subset) 
            image_archive = 'Potsdam/4_Ortho_RGB.zip'
            mask_archive = 'Potsdam/5_Labels_all.zip'

            def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
                return (
                    DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                    .merge(spatial_df, how = 'left', left_index=True, right_index=True)
                    .assign(x_off = lambda df: df.apply(lambda col: col["x_off"] + col["tile_tl_0"] * col["x_res"], axis = 1))
                    .assign(y_off = lambda df: df.apply(lambda col: col["y_off"] + col["tile_tl_0"] * col["y_res"], axis = 1))
                    .assign(image_height = tile_size[0])
                    .assign(image_width = tile_size[1])
                    .reset_index(drop = True)
                )

            if to == "imagefolder":
                imagefolder_path = cls.local / "imagefolder" / subset
                image_dir_path = fs.get_new_dir(imagefolder_path, "images")
                mask_dir_path = fs.get_new_dir(imagefolder_path, "masks")

                with zipfile.ZipFile(archive_path, mode = 'r') as zf_outer:
                    if tile_size is not None and tile_stride is not None:
                        tiled_df = _tile(index_df, spatial_df)

                        with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                             for image_path in tqdm(tiled_df["image_path"].unique()):
                                 image = iio.imread(zf_inner.read(f"4_Ortho_RGB/{image_path}"), extension=".tif")
                                 for _, row in tiled_df[tiled_df["image_path"] == image_path].iterrows():
                                     tile_kwargs = {
                                         "fp": str(image_dir_path/f"{image_path}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif"),
                                         "mode": "w", "driver": "Gtiff", "dtype": np.uint8, "count": 3, 
                                         "crs": CRS.from_epsg(row["crs"]), "width": tile_size[0], "height": tile_size[1],
                                         "transform": Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                                     }
                                     with rio.open(**tile_kwargs) as raster:
                                         raster.write(cls.read_tile(image, row).transpose(2,0,1))

                        with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                            for mask_path in tqdm(tiled_df["mask_path"].unique()):
                                mask = iio.imread(zf_inner.read(mask_path), extension=".tif")
                                mask = np.clip(np.where(mask[:, :, 0] == 255, 0, 255) - mask[:, :, 1], 0, 255)
                                for _, row in tiled_df[tiled_df["mask_path"] == mask_path].iterrows():
                                    tile_kwargs = {
                                        "fp": str(mask_dir_path/f"{mask_path}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif"),
                                        "mode": "w", "driver": "Gtiff", "dtype": np.uint8, "count": 1, 
                                        "crs": CRS.from_epsg(row["crs"]), "width": tile_size[0], "height": tile_size[1],
                                        "transform": Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                                    }
                                    with rio.open(**tile_kwargs) as raster:
                                        raster.write(cls.read_tile(mask, row)[np.newaxis, :, :])

                    else:
                        with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                            for image_path in tqdm([x for x in zf_inner.namelist() if x.startswith("top/") and x.endswith(".tif")]):
                                zf_inner.extract(image_path, image_dir_path/image_path.split('/')[-1])

                        with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                            for mask_path in tqdm(zf_inner.namelist()):
                                zf_inner.extract(mask_path, mask_dir_path/mask_path)

            elif to == "hdf5":
                hdf5_path = cls.local / "hdf5" / f"{subset}.h5"

                with zipfile.ZipFile(archive_path, mode = 'r') as zf_outer:
                    with h5py.File(hdf5_path, mode = 'w') as f:
                        if tile_size is not None and tile_stride is not None:
                            tiled_df = _tile(index_df, spatial_df)
                            images = f.create_dataset('images', (len(tiled_df), *tile_size, 4))
                            masks = f.create_dataset('masks', len(tiled_df), dtype = h5py.special_dtype(vlen=np.uint8))
                            bg_masks = f.create_dataset('bg_masks', len(tiled_df), dtype = h5py.special_dtype(vlen=np.uint8))

                            with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                                for image_path in tqdm(tiled_df["image_path"].unique()):
                                    image = iio.imread(zf_inner.read(f"4_Ortho_RGB/{image_path}"), extension=".tif")
                                    for idx, row in tiled_df[tiled_df["image_path"] == image_path].iterrows():
                                        images[idx] = cls.read_tile(image, row)

                            with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                                for mask_path in tqdm(tiled_df["image_path"].unique()):
                                    mask = iio.imread(zf_inner.read(mask_path), extension=".tif")
                                    for _, row in tiled_df[tiled_df["image_path"] == mask_path].iterrows():
                                        mask_tile = cls.read_tile(mask, row)
                                        masks[idx] = cls.encode_binary_mask(mask_tile[:, : ,1].squeeze())
                                        bg_masks[idx] = cls.encode_binary_mask(mask_tile[:, :, 0].squeeze())
    
                            tiled_df[index_df.columns].to_hdf(hdf5_path, key = "index", mode = "r+")
                            tiled_df[spatial_df.columns].to_hdf(hdf5_path, key = "spatial", mode = "r+")

                        else:
                            images = f.create_dataset('images', (len(index_df), 6000, 6000, 4), dtype = h5py.special_dtype(vlen=np.uint8))
                            masks = f.create_dataset('masks', len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))
                            bg_masks = f.create_dataset('bg_masks', len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))

                            with zipfile.ZipFile(BytesIO(zf_outer.read(image_archive))) as zf_inner:
                                for idx, row in tqdm(index_df.iterrows()):
                                    images[idx] = iio.imread(zf_inner.read(f"top/{image_path}"), extension='.tif').flatten()

                            with zipfile.ZipFile(BytesIO(zf_outer.read(mask_archive))) as zf_inner:
                                for idx, row in tqdm(index_df.iterrows()):
                                    mask = iio.imread(zf_inner.read(image_path), extension='.tif')
                                    masks[idx] = cls.encode_binary_mask(mask[:, :, 1]).squeeze() 
                                    bg_masks[idx] = cls.encode_binary_mask(mask[:, :, 0].squeeze())
                            index_df.to_hdf(hdf5_path, key = "index", mode = "r+")
                            spatial_df.to_hdf(hdf5_path, key = "spatial", mode = "r+")

    @staticmethod
    def encode_binary_mask(mask : NDArray) -> NDArray:
        # NOTE: might have to change 255 to mask.max() to adapt to more bands
        mask = np.where(mask.flatten() == 255, 1, 0)
        indices = np.nonzero(np.diff(mask, n = 1))[0] + 1
        indices = np.concat(([0], indices, [len(mask)]))
        rle = np.diff(indices, n = 1)
        if mask[0] == 1:
            return np.concat(([0], rle))
        return rle

    @staticmethod
    def decode_binary_mask(rle: NDArray, shape: tuple):
        mask = np.zeros(rle.sum(), dtype = np.uint8)
        idx, fill = 0, 0
        for run in rle:
            new_idx = idx + run
            if fill == 1:
                mask[idx: new_idx] = 1 
                idx, fill = new_idx, 0
            else:
                idx, fill = new_idx, 1
        return mask.reshape(shape)

    @staticmethod
    def read_tile(scene: NDArray, row: pd.Series) -> NDArray:
        def _pad(n: int) -> tuple[int, int]:
            return (n//2, n//2) if n % 2 == 0 else (n//2, (n//2) + 1)

        tile = scene[row["tile_tl_0"] : min(row["tile_br_0"], 5000), row["tile_tl_1"] : min(row["tile_br_1"], 5000)]

        if scene.ndim == 3:
            tile_size = (row["tile_br_0"] - row["tile_tl_0"], row["tile_br_1"] - row["tile_tl_1"], scene.shape[2])
        elif scene.ndim == 2:
            tile_size = (row["tile_br_0"] - row["tile_tl_0"], row["tile_br_1"] - row["tile_tl_1"])

        if tile.shape != tile_size: 
            pad_width = [_pad(tile_size[0] - tile.shape[0]), _pad(tile_size[1] - tile.shape[1]), (0, 0)]
            tile = np.pad(array = tile, pad_width = (pad_width[:2] if tile.ndim == 2 else pad_width))
        assert tile.shape == tile_size
        return tile

ISPRSIndexSchema = pa.DataFrameSchema(
    columns={
        "image_path": pa.Column(str, coerce=True),
        "mask_path": pa.Column(str, coerce=True),
        "location": pa.Column(str, coerce=True),
        "split": pa.Column(str, pa.Check.isin(Dataset.valid_splits))
    },
    index=pa.Index(int, unique=True)
)     

class Vaihingen_Segmentation_Imagefolder(Dataset): ...
class Vaihingen_Segmentation_HDF5(Dataset): ...
class Potsdam_Segmentation_Imagefolder(Dataset): ...
class Potsdam_Segmentation_HDF5(Dataset): ...