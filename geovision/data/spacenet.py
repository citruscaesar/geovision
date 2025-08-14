from typing import Any, Literal, Optional, Callable, Sequence
from numpy.typing import NDArray

import os
import h5py
import py7zr
import torch
import shutil
import tarfile 
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
from rasterio.features import sieve, shapes
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url
from skimage.measure import find_contours, approximate_polygon

from geovision.data import Dataset, DatasetConfig
from geovision.data.transforms import SegmentationCompose

class SpaceNet:
    local = Path.home() / "datasets" / "spacenet"

    subsets: dict[str, tuple[str, ...]] = {
        "rio": ("buildings",),
        "vegas": ("buildings", "roads"),
        "paris": ("buildings", "roads"),
        "shanghai": ("buildings", "roads"),
        "khartoum": ("buildings", "roads"),
        "atlanta": ("buildings",),
        "moscow": ("roads",),
        "mumbai": ("roads",),
        "san_juan": ("roads",),
        "dar_es_salaam": ("roads",),
        "rotterdam": ("buildings",), # MS + SAR
        "sn7": ("buildings",), # Time Series
        "new_orleans": ("buildings", "roads"), # Flooded Conditions
        "dernau": ("buildings", "roads"), # Flooded Conditions
    }

    @classmethod
    def extract(
        cls, 
        subset: Literal["rio", "vegas", "paris", "shanghai", "khartoum", "atlanta", "moscow", "mumbai", "san_juan", "dar_es_salaam", "rotterdam", 
                        "sn7", "new_orleans", "dernau"],
        labels: Literal["buildings", "roads"],
    ):
        AOIs = {
            "buildings": {
                "rio": {
                    #'SN1_buildings_test_AOI_1_Rio_3band': "rio_buildings_test", # No Labels
                    'SN1_buildings_train_AOI_1_Rio_geojson_buildings': "rio_buildings_geojson",
                    'SN1_buildings_train_AOI_1_Rio_3band': "rio_buildings_train",
                    #'SN1_buildings_train_AOI_1_Rio_metadata': "rio_metadata", # Redundant
                },
             
                "vegas": {
                    'SN2_buildings_train_AOI_2_Vegas': "vegas_buildings_train",
                    'AOI_2_Vegas_Test_public': "vegas_buildings_test",
                },

                "paris": {
                    'AOI_3_Paris_Test_public': "paris_buildings_test",
                    'SN2_buildings_train_AOI_3_Paris': "paris_buildings_train",
                },

                "shanghai": {
                    'AOI_4_Shanghai_Test_public': "shanghai_buildings_test",
                    'SN2_buildings_train_AOI_4_Shanghai': "shanghai_buildings_train",
                },

                "khartoum": {
                    'AOI_5_Vegas_Test_public': "vegas_buildings_test",
                    'SN2_buildings_train_AOI_5_Vegas': "vegas_buildings_train",
                },

                "rotterdam": {
                    'SN6_buildings_AOI_11_Rotterdam_test_public': "rotterdam_buildings_test",
                    'SN6_buildings_AOI_11_Rotterdam_train': "rotterdam_buildings_train"
                },
            
                "sn7": {
                    "SN7_buildings_test_public": "sn7_buildings_test",
                    "SN7_buildings_train": "sn7_buildings_train",
                    "SN7_buildings_train_csvs": "sn7_buildings_train_csvs"
                },
                
                "atlanta": dict(),

                "new_orleans": {
                    "Louisiana-East_Training_Public": "new_orleans_buildings_train",
                    "Louisiana-West_Test_Public": "new_orleans_buildings_test",
                }

            },

            "roads": {
                "vegas": {
                    'SN3_roads_train_AOI_2_Vegas': "vegas_roads_train",
                    'SN3_roads_test_public_AOI_2_Vegas': "vegas_roads_test",
                    'SN3_roads_train_AOI_2_Vegas_geojson_roads_speed': "vegas_roads",
                },

                "paris": {
                    'SN3_roads_train_AOI_3_Paris_geojson_roads_speed': "paris_roads",
                    'SN3_roads_train_AOI_3_Paris': "paris_roads_train",
                    'SN3_roads_test_public_AOI_3_Paris': "paris_roads_test",
                },


                "shanghai": {
                    'SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed': "shanghai_roads",
                    'SN3_roads_train_AOI_4_Shanghai': "shanghai_roads_train",
                    'SN3_roads_test_public_AOI_4_Shanghai': "shanghai_roads_test",
                },

                "khartoum": {
                    'SN3_roads_train_AOI_5_Vegas_geojson_roads_speed': "vegas_roads",
                    'SN3_roads_train_AOI_5_Vegas': "vegas_roads_train",
                    'SN3_roads_test_public_AOI_5_Vegas': "vegas_roads_test",
                },

                "moscow": {
                    "SN5_roads_train_AOI_7_Moscow": "moscow_roads_train",
                    "SN5_roads_test_public_AOI_7_Moscow": "moscow_roads_test"
                },

                "mumbai": {
                    "SN5_roads_train_AOI_8_Mumbai": "mumbai_roads_train",
                    "SN5_roads_test_public_AOI_8_Mumbai": "mumbai_roads_test"
                },

                "san_juan": {
                    "SN5_roads_test_public_AOI_9_San_Juan": "san_juan_roads_test"
                }
            }
        }

        assert subset in cls.subsets
        if subset == "rio":
            challenge = "SN1_buildings"
        elif subset in ("vegas", "paris", "shanghai", "khartoum"):
            challenge = "SN2_buildings" if labels == "buildings" else "SN3_roads"
        elif subset == "atlanta":
            challenge = "SN4_buildings"
        elif subset in ("moscow", "mumbai", "san_juan"):
            challenge = "SN5_roads"
        elif subset == "rotterdam":
            challenge = "SN6_buildings"
        elif subset == "sn7":
            challenge = "SN7_buildings"
        elif subset in ("new_orleans", "dernau"):
            challenge = "SN8_buildings"

        remote_prefix = "s3://spacenet-dataset/spacenet"
        if os.environ.get("S3_ENDPOINT_URL") is not None:
            del os.environ["S3_ENDPOINT_URL"]
        
        commands = "" 
        for k,v in AOIs[labels][subset].items():
            commands += f"cp --sp {remote_prefix}/{challenge}/tarballs/{k}.tar.gz {fs.get_new_dir(cls.local/"staging"/subset)}/{v}.tar.gz\n"
        #print(commands[:-1])
        subprocess.run(["s5cmd", "--no-sign-request", "run"], input = commands[:-1].encode())

    @classmethod
    def download(cls, subset): ...
    # download from personal s3://spacenet to cls.local/hdf5

    @classmethod
    def load(
        cls, 
        table: Literal["index", "spatial"],
        src: Literal["archive", "imagefolder", "hdf5"],
        subset: Literal["rio", "vegas", "paris", "shanghai", "khartoum", "atlanta", "moscow", "mumbai", "san_juan", "dar_es_salaam", "rotterdam", 
                        "sn7", "new_orleans", "dernau"],
        labels: Literal["buildings", "roads"] 
    ) -> pd.DataFrame: 

        def add_metadata(raster: rio.DatasetReader, metadata: dict[str, Any]):
            metadata["image_width"].append(raster.width)
            metadata["image_height"].append(raster.height)
            metadata["x_off"].append(raster.transform.xoff)
            metadata["y_off"].append(raster.transform.yoff)
            metadata["x_res"].append(raster.transform.a)
            metadata["y_res"].append(raster.transform.e)
            metadata["crs"].append(raster.crs.to_epsg())

        assert src in ("archive", "imagefolder", "hdf5")
        assert subset in cls.subsets.keys()
        assert labels in cls.subsets[subset] 

        if src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"spacenet_{subset}.h5", key = table, mode = "r")

        if subset == "rio":
            if src == "archive":
                archive_path = cls.local / "staging" / f"{subset}"
                try: 
                    return pd.read_hdf(archive_path / "metadata.h5", key = table, mode = "r")
                except OSError:
                    assert table in ("index", "spatial")
                    metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}

                    train_archive_path = archive_path / f"{subset}_buildings_train.tar.gz"
                    with tarfile.open(train_archive_path, mode= "r") as tf:
                        for image_path in tqdm(tf.getnames()[1:]):
                            metadata["image_path"].append(image_path)
                            with rio.open(rio.MemoryFile(tf.extractfile(image_path))) as raster:
                                add_metadata(raster, metadata)
                    
                    df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop = True)
                    df["location"] = subset
                    df["mask_path"] = df["image_path"].apply(lambda x: x.replace(".tif", ".geojson").replace("3band/3band", "geojson/Geo"))

                    index_df = df[["image_path", "mask_path", "location"]]
                    spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                    index_df.to_hdf(archive_path / "metadata.h5", key = "index", mode = 'a', index = True)
                    spatial_df.to_hdf(archive_path / "metadata.h5", key = "spatial", mode = 'r+', index = True)
                    return index_df if table == "index" else spatial_df
            
            elif src == "imagefolder":
                imagefolder_path = cls.local / "imagefolder" / f"{subset}"
                try: 
                    return pd.read_hdf(imagefolder_path / "metadata.h5", key = table, mode = "r")
                except OSError:
                    assert table in ("index", "spatial")
                    metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}
                    for image_path in tqdm((imagefolder_path / "images").iterdir()):
                        metadata["image_path"].append(f"images/{image_path.name}")
                        with rio.open(image_path) as raster:
                            add_metadata(raster, metadata)

                    df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop = True)
                    df["location"] = subset
                    df["mask_path"] = df["image_path"].apply(
                        lambda x: x.replace("images", "masks").replace(".tif", ".geojson").replace("3band", "Geo")
                    )

                    index_df = df[["image_path", "mask_path", "location"]]
                    spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                    index_df.to_hdf(imagefolder_path / "metadata.h5", key = "index", mode = 'a', index = True)
                    spatial_df.to_hdf(imagefolder_path / "metadata.h5", key = "spatial", mode = 'r+', index = True)
                    return index_df if table == "index" else spatial_df
        
        elif subset in ("paris", "vegas", "shanghai", "khartoum"):
            if src == "archive":
                archive_dir = cls.local / "staging" / f"{subset}"
                try: 
                    return pd.read_hdf(archive_dir / "metadata.h5", key = f"{table}", mode = "r")
                except OSError:
                    assert table in ("index", "spatial")
                    metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}

                    train_archive_path = archive_dir / f"{subset}_buildings_train.tar.gz"
                    with tarfile.open(train_archive_path, mode= "r") as tf:
                        # Roads:  PS-RGB/*.tif , geojson-roads/*.geojson
                        # Buildings:  
                        for image_path in tqdm(sorted(tf.getnames()[1:])):
                            metadata["image_path"].append(f"{train_archive_path}!/{image_path}")
                            with rio.open(rio.MemoryFile(tf.extractfile(image_path))) as raster:
                                add_metadata(raster, metadata)

                    test_archive_path = archive_dir/f"{subset}_buildings_test.tar.gz"
                    with tarfile.open(test_archive_path, mode= "r") as tf:
                        for image_path in tqdm(sorted(tf.getnames()[1:])):
                            metadata["image_path"].append(f"{test_archive_path}!/{image_path}")
                            with rio.open(rio.MemoryFile(tf.extractfile(image_path))) as raster:
                                add_metadata(raster, metadata)

                    df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop = True)
                    df["location"] = subset

                    index_df = df[["image_path", "location"]]
                    spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                    index_df.to_hdf(archive_dir/"metadata.h5", key = "index", mode = 'a', index = True)
                    spatial_df.to_hdf(archive_dir/"metadata.h5", key = "spatial", mode = 'r+', index = True)

                    return index_df if table == "index" else spatial_df

        elif subset == "rotterdam":
            if src == "archive":
                archive_dir = cls.local / "staging" / f"{subset}"
                try: 
                    return pd.read_hdf(archive_dir / "metadata.h5", key = table, mode = "r")
                except OSError:
                    assert table in ("index", "spatial")
                    #metadata = {k: list() for k in ("image_path", "image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs")}

                    # Buildings:  PS-RGBNIR/*.tif, SAR-Intensity/*.tif , geojson-buildings/*.geojson
                    train_archive_path = archive_dir / f"{subset}_buildings_train.tar.gz"
                    with tarfile.open(train_archive_path, mode= "r") as tf:
                        train_files = tf.getnames()
                    
                    images_df = (
                        pd.DataFrame({"image_path": [x for x in train_files if x.endswith(".tif")]})
                        .assign(image_path = lambda df: df["image_path"].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[-4:])))
                        .assign()
                    )
                    
                    index_df = pd.DataFrame({"image_path": train_files})
                    index_df.to_hdf(archive_dir/"metadata.h5", key = "index", mode = "w", complevel=9, complib="zlib")
                    return index_df
                        # for image_path in tqdm(sorted(tf.getnames()[1:])):
                            # metadata["image_path"].append(f"{train_archive_path}!/{image_path}")
                            # with rio.open(rio.MemoryFile(tf.extractfile(image_path))) as raster:
                                # add_metadata(raster, metadata)

                    # test_archive_path = archive_dir/f"{subset}_buildings_test.tar.gz"
                    # with tarfile.open(test_archive_path, mode= "r") as tf:
                        # for image_path in tqdm(sorted(tf.getnames()[1:])):
                            # metadata["image_path"].append(f"{test_archive_path}!/{image_path}")
                            # with rio.open(rio.MemoryFile(tf.extractfile(image_path))) as raster:
                                # add_metadata(raster, metadata)

                    # df = pd.DataFrame(metadata).sort_values("image_path").reset_index(drop = True)
                    # df["location"] = subset

                    # index_df = df[["image_path", "location"]]
                    # spatial_df = df[["image_height", "image_width", "x_off", "y_off", "x_res", "y_res", "crs"]]
                    # index_df.to_hdf(archive_dir/"metadata.h5", key = "index", mode = 'a', index = True)
                    # spatial_df.to_hdf(archive_dir/"metadata.h5", key = "spatial", mode = 'r+', index = True)

                    # return index_df if table == "index" else spatial_df

        elif subset == "atlanta": ...
        elif subset == "moscow": ...
        elif subset == "mumbai": ...
        elif subset == "san_juan": ...
        elif subset == "dar_es_salaam": ...
        elif subset == "sn7": ...
        elif subset == "new_orleans": ...
        elif subset == "dernau": ...

    @classmethod
    def transform(
        cls,
        to: Literal["imagefolder", "hdf5"], 
        subset: Literal["rio", "vegas", "paris", "shanghai", "khartoum", "atlanta", "moscow", "mumbai", "san_juan", "dar_es_salaam", "rotterdam", 
                        "sn7", "new_orleans", "dernau"],
        labels: Literal["buildings", "roads"],
        tile_size: Optional[int | tuple[int, int]] = None,
        tile_stride: Optional[int | tuple[int, int]] = None,
    ):
        assert to in ("imagefolder", "hdf5")
        assert subset in cls.subsets.keys()
        assert labels in cls.subsets[subset]

        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            assert isinstance(tile_size, Sequence) and len(tile_size) == 2

        elif tile_stride is not None:
            if isinstance(tile_stride, int):
                tile_stride = (tile_stride, tile_stride)
            assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            #index_columns = index_df.columns
            return (
                DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                .merge(spatial_df, how = 'left', left_index=True, right_index=True)
                .assign(x_off = lambda df: df.apply(lambda col: col["x_off"] + col["tile_tl_0"] * col["x_res"], axis = 1))
                .assign(y_off = lambda df: df.apply(lambda col: col["y_off"] + col["tile_tl_0"] * col["y_res"], axis = 1))
                .assign(image_height = tile_size[0])
                .assign(image_width = tile_size[1])
                .reset_index(drop = True)
            )

        image_archive_path = cls.local / "staging" / subset / f"{subset}_{labels}_train.tar.gz"
        label_archive_path = cls.local / "staging" / subset / f"{subset}_{labels}_geojson.tar.gz"
        if to == "imagefolder":
            imagefolder_path = cls.local / "imagefolder" / subset
            
            if tile_stride is not None and tile_size is not None:
                fs.get_new_dir(imagefolder_path / "images")
                fs.get_new_dir(imagefolder_path / "masks")

                index_df = cls.load('index', 'archive', subset, labels)
                spatial_df = cls.load('spatial', 'archive', subset, labels)
                tiled_df = _tile(index_df, spatial_df)

                # For Each (Original Scene) Image
                with tarfile.open(image_archive_path) as image_tf:
                    with tarfile.open(label_archive_path) as label_tf:
                        for image_path in tqdm(tiled_df["image_path"].unique()):
                            # Filter and Load Image 
                            df = tiled_df[tiled_df["image_path"] == image_path]

                            with rio.open(image_tf.extractfile(df["image_path"].iloc[0])) as raster:
                                image = raster.read().transpose(1,2,0)
                                image_meta = raster.meta

                            mask_df = gpd.read_file(BytesIO(label_tf.extractfile(df["mask_path"].iloc[0]).read()))
                            mask = rio.features.rasterize(
                                shapes = [(g, 255) for g in mask_df.geometry], 
                                out_shape = (image_meta["height"], image_meta["width"]), 
                                transform = image_meta["transform"], 
                                all_touched = True,
                                dtype = np.uint8
                            )
                    
                            kwargs = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'height': tile_size[0], 'width': tile_size[1]}

                            for _, row in df.iterrows():
                                filename = f"{row["image_path"].split('/')[-1].removesuffix(".tif")}_{row["tile_tl_0"]}_{row["tile_tl_1"]}.tif"

                                tile_kwargs = kwargs.copy()
                                tile_kwargs["fp"] = imagefolder_path / "images" / filename
                                tile_kwargs["count"] = 3
                                tile_kwargs["crs"] = CRS.from_epsg(row["crs"])
                                tile_kwargs["transform"] = Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                                with rio.open(**tile_kwargs, mode = 'w') as raster:
                                    raster.write(cls.read_tile(image, row).transpose(2,0,1))

                                tile_kwargs["fp"] = Path(str(tile_kwargs["fp"]).replace("images", "masks").replace("3band", "Geo"))
                                tile_kwargs["count"] = 1
                                with rio.open(**tile_kwargs, mode = 'w') as raster:
                                    raster.write(cls.read_tile(mask, row)[np.newaxis,:,:])

            else:
                with tarfile.open(image_archive_path) as tf:
                    tf.extractall(imagefolder_path)
                shutil.move(imagefolder_path / "3band", imagefolder_path / "images")

                mask_dir = fs.get_new_dir(imagefolder_path / "masks")
                with tarfile.open(label_archive_path) as tf:
                    for image_path in tqdm(list((imagefolder_path / "images").iterdir())[3182:]):
                        with rio.open(image_path) as raster:
                            kwargs = raster.meta
                            kwargs["count"] = 1

                        mask_path = mask_dir / image_path.name.replace("3band", "Geo")
                        mask_df = gpd.read_file(BytesIO(tf.extractfile(f"geojson/{mask_path.stem}.geojson").read()))
                        mask = rio.features.rasterize(
                            shapes = [(g, 255) for g in mask_df.geometry], 
                            out_shape = (kwargs["height"], kwargs["width"]), 
                            transform = kwargs["transform"], 
                            all_touched = True,
                            dtype = np.uint8
                        )

                        with rio.open(fp = mask_path, mode = 'w', **kwargs) as raster:
                            raster.write(mask[np.newaxis,:,:])

        if to == "hdf5":
            hdf5_path = cls.local / "hdf5" / f"{subset}.h5" 
            index_df = cls.load('index', 'archive', subset, labels)
            spatial_df = cls.load('spatial', 'archive', subset, labels)

            if tile_size is not None and tile_stride is not None:
                tiled_df = _tile(index_df, spatial_df)
                with h5py.File(hdf5_path) as f:
                    images = f.create_dataset("images", (len(tiled_df, *tile_size, 3)), dtype = np.uint8)
                    masks = f.create_dataset("masks", len(tiled_df), dtype = h5py.special_dtype(vlen = np.uint8))

                    with tarfile.open(image_archive_path) as image_tf:
                        with tarfile.open(label_archive_path) as label_tf:
                            for image_path in tqdm(tiled_df["image_path"].unique()):
                                # Filter and Load Image 
                                df = tiled_df[tiled_df["image_path"] == image_path]

                                with rio.open(image_tf.extractfile(df["image_path"].iloc[0])) as raster:
                                    image = raster.read().transpose(1,2,0)
                                    image_meta = raster.meta

                                mask_df = gpd.read_file(BytesIO(label_tf.extractfile(df["mask_path"].iloc[0]).read()))
                                mask = rio.features.rasterize(
                                    shapes = [(g, 255) for g in mask_df.geometry], 
                                    out_shape = (image_meta["height"], image_meta["width"]), 
                                    transform = image_meta["transform"], 
                                    all_touched = True, 
                                    dtype = np.uint8
                                )

                                for i, row in df.iterrows():
                                    images[i] = cls.read_tile(image, row)
                                    masks[i] = cls.encode_binary_mask(cls.read_tile(mask, row))

                    tiled_df["image_path"] = tiled_df.apply(lambda col: f"{col["image_path"].split('/')[-1].removesuffix('.tif')}_{col["tile_tl_0"]}_{col["tile_tl_1"]}.tif", axis = 1)
                    tiled_df[["image_path", "location"]].to_hdf(hdf5_path, key = 'index', mode = 'r+')
                    tiled_df.drop(columns=["image_path", "mask_path", "location", "index"]).to_hdf(hdf5_path, key = 'spatial', mode = 'r+')
            else: 
                # images sizes are inconsistent
                with tarfile.open(image_archive_path) as image_tf:
                    with tarfile.open(label_archive_path) as label_tf:
                        with h5py.File(hdf5_path, mode = 'w') as f:
                            images = f.create_dataset("images", len(index_df), dtype = h5py.special_dtype(vlen = np.uint8))
                            masks = f.create_dataset("masks", len(index_df), dtype = h5py.special_dtype(vlen = np.uint8))

                            for idx, row in tqdm(index_df.iterrows(), total = len(index_df)):
                                with rio.open(image_tf.extractfile(row["image_path"])) as raster:
                                    images[idx] = np.frombuffer(iio.imwrite("<bytes>", raster.read().transpose(1,2,0), extension=".jpg"), dtype = np.uint8)
                                    image_meta = raster.meta

                                mask_df = gpd.read_file(BytesIO(label_tf.extractfile(df["mask_path"]).read()))
                                masks[idx] = cls.encode_binary_mask(rio.features.rasterize(
                                    shapes = [(g, 255) for g in mask_df.geometry], 
                                    out_shape = (raster.meta["height"], raster.meta["width"]), 
                                    transform = raster.meta["transform"], 
                                    all_touched = True, 
                                    dtype = np.uint8
                                ))





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
        tile_size = (row["tile_br_0"] - row["tile_tl_0"], row["tile_br_1"] - row["tile_tl_1"], 3)
        if scene.ndim == 2:
            tile_size = tile_size[:2] 

        if tile.shape != tile_size: 
            pad_width = [_pad(tile_size[0] - tile.shape[0]), _pad(tile_size[1] - tile.shape[1]), (0, 0)]
            tile = np.pad(array = tile, pad_width = (pad_width[:2] if tile.ndim == 2 else pad_width))
        return tile