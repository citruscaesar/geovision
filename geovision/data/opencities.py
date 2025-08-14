from typing import Literal, Optional, Sequence
from numpy.typing import NDArray

import os 
import h5py
import torch
import shutil
import shapely
import subprocess
import numpy as np
import pandas as pd
import pandera as pa
import rasterio as rio
import geopandas as gpd
import imageio.v3 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from rasterio import features
from skimage.transform import rescale
from torchvision.transforms import v2 as T

from geovision.data import Dataset, DatasetConfig
from geovision.io.local import FileSystemIO as fs

class OpenCities:
    local = Path.home() / "datasets" / "opencities"
    class_names = ("background", "building") 
    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="stratified",
        tabular_sampler_params={"split_on": "location", "val_frac": 0.15, "test_frac": 0.15},
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=T.Compose([T.RandomCrop(128, pad_if_needed=True), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.RandomResizedCrop(128)
    )

    location_code = {
        "acc": "Accra",
        "kam": "Kampala",
        "ptn": "Pointe-Noire",
        "znz": "Zanzibar",
        "gao": "Ngaoundere",
        "mah": "Mahe Island",
        "dar": "Dar Es Salaam",

        "mon": "Mon?",
        "kin": "Kinshasa",
        "nia": "Niamey"
    }

    @classmethod
    def extract(): ...

    @classmethod
    def download(): ...

    @classmethod
    def load(
        cls, 
        table: Literal["index", "spatial", "spectral", "temporal", "radiometric"],
        src: Literal["staging", "imagefolder", "hdf5"],
        subset: Literal["opencities", "opencities_tier1", "opencities_tier2", "opencities_unlabelled"]
    ):
        assert src in ("staging", "imagefolder", "hdf5")
        assert subset in ("opencities", "opencities_tier1", "opencities_tier2", "opencities_unlabelled") 

        def _get_label_name(x: str) -> str | None:
            parts = x.split('/')
            if parts[0] == "test":
                return None
            parts[-1] = parts[-1].replace(".tif", ".geojson")
            parts[-2] = parts[-2] + "-labels"
            return '/'.join(parts)

        def _get_location(x: str) -> str | None:
            parts = x.split('/')
            if parts[0] == "test":
                return None
            return cls.location_code[parts[-3]]

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/f"{subset}.h5", key=table, mode="r")

        if src == "staging":
            staging_path = fs.get_valid_dir_err(cls.local/"staging", empty_ok=False)
            try:
                raise OSError
                return pd.read_hdf(staging_path/"metadata.h5", key=table, mode="r")
            except OSError:
                assert table in ("index", "spatial")
                index_df = (
                    pd.DataFrame({"image_path": sorted(staging_path.rglob("*.tif"))})
                    .assign(image_path = lambda df: df["image_path"].apply(lambda x: '/'.join(x.as_posix().split('/')[-4:]).removeprefix("staging/")))
                    .assign(label_path = lambda df: df["image_path"].apply(_get_label_name))
                    .assign(location = lambda df: df["image_path"].apply(_get_location))
                    .assign(subset = lambda df: df["image_path"].str.split('/', expand=True)[0])
                )           
                index_df = index_df[index_df["subset"] != "test"].reset_index(drop=True)

                # spatial_info, corrupt_idxs = list(), set() 
                # for idx, row in tqdm(index_df.iterrows(), total = len(index_df), desc = "reading metadata... "):
                    # # check label
                    # try:
                        # if row["label_path"] is not None:
                            # gpd.read_file(staging_path/row["label_path"])
                    # except Exception as e:
                        # corrupt_idxs.add(idx)
                        # print(f"Error reading labels for idx = {idx} [{e}]")
                        # continue

                    # # check raster 
                    # with rio.open(staging_path/row["image_path"]) as raster:
                        # #try:
                            # #for _, window in raster.block_windows():
                                # #raster.read(window=window)
                        # #except Exception as e:
                            # #print(f"Error reading raster for idx = {idx} [{e}: {staging_path/row["image_path"]}]")
                            # #corrupt_idxs.add(idx)
                            # #continue
                        # transform = raster.transform

                    # # if all goes well
                    # spatial_info.append((raster.height, raster.width, raster.crs.to_epsg(), transform.a, transform.e, transform.xoff, transform.yoff))

                #index_df = index_df.drop(index=corrupt_idxs).reset_index(drop=True)
                #spatial_df = pd.DataFrame(spatial_info, index_df.index, ["image_height", "image_width", "crs", "x_res", "y_res", "x_off", "y_off"])

                index_df.to_hdf(staging_path/"metadata.h5", key="index", mode="w", format="fixed", complib="zlib", complevel=9)
                #spatial_df.to_hdf(staging_path/"metadata.h5", key="spatial", mode="a", format="fixed", complib="zlib", complevel=9)
                
                if table == "index":
                    return index_df
                # elif table == "spatial":
                    # return spatial_df
    
    @classmethod
    def transform(
            cls,
            to: Literal["imagefolder", "hdf5"], 
            subset: Literal["opencities", "opencities_tier_1", "opencities_tier_2", "opencities_unlabelled"],
            tile_size: Optional[int | tuple[int, int]] = None,
            tile_stride: Optional[int | tuple[int, int]] = None,
            to_jpg: bool = False, 
            jpg_quality: int = 95, 
            chunk_size: Optional[int] = None,
            downsampling_factor: Optional[int] = None,
        ):
        assert to in ("imagefolder", "hdf5")
        assert subset in ("opencities", "opencities_tier_1", "opencities_tier_2", "opencities_unlabelled") 

        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            assert isinstance(tile_size, Sequence) and len(tile_size) == 2

        elif tile_stride is not None:
            if isinstance(tile_stride, int):
                tile_stride = (tile_stride, tile_stride)
            assert isinstance(tile_stride, Sequence) and len(tile_stride) == 2

        if to_jpg:
            assert isinstance(jpg_quality, int) and jpg_quality <= 95 and jpg_quality > 0

        if chunk_size is not None:
            assert isinstance(chunk_size, int)

        if downsampling_factor is not None:
            assert isinstance(downsampling_factor, int) and downsampling_factor >= 1

        def _tile(index_df: pd.DataFrame, spatial_df: pd.DataFrame) -> pd.DataFrame:
            return (
                DatasetConfig.sliding_window_spatial_sampler(index_df, spatial_df, tile_size, tile_stride)
                .merge(spatial_df, how = 'left', left_index=True, right_index=True)
                .assign(x_off = lambda df: df.apply(lambda col: col["x_off"] + col["tile_tl_0"] * col["x_res"], axis = 1))
                .assign(y_off = lambda df: df.apply(lambda col: col["y_off"] + col["tile_tl_0"] * col["y_res"], axis = 1))
                #.assign(image_height = tile_size[0])
                #.assign(image_width = tile_size[1])
                .reset_index(drop = True)
            )
        
        staging_path = cls.local/"staging"
        
        index_df = cls.load('index', 'staging', 'opencities')
        spatial_df = cls.load('spatial', 'staging', 'opencities')

        corrupt_images_filter = index_df["image_path"].apply(lambda x: not x.endswith("665946.tif"))
        index_df = index_df[corrupt_images_filter]
        spatial_df = spatial_df[corrupt_images_filter]

        if subset != "opencities":
            if subset == "opencities_unlabelled":
                drop_idxs = index_df[index_df["subset"] != "test"].index
            elif subset == "opencities_tier_1":
                drop_idxs = index_df[index_df["subset"] != "train_tier_1"].index
            elif subset == "opencities_tier_2":
                drop_idxs = index_df[index_df["subset"] != "train_tier_2"].index

            # display(index_df)
            # display(spatial_df)
            # display(drop_idxs)
            index_df = index_df.drop(index=drop_idxs).reset_index(drop=True)
            spatial_df = spatial_df.drop(index=drop_idxs).reset_index(drop=True)

        if to == "imagefolder":
            imagefolder_path = cls.local/"imagefolder"
            images_dir = fs.get_new_dir(imagefolder_path, subset, "images")
            masks_dir = fs.get_new_dir(imagefolder_path, subset, "masks")

            if tile_size is not None and tile_stride is not None:
                tiled_df = (
                    _tile(index_df, spatial_df).set_index("image_path")
                    .rename(columns={"tile_tl_0": "tile_y_min", "tile_tl_1": "tile_x_min", "tile_br_0": "tile_y_max", "tile_br_1": "tile_x_max"})
                    .assign(tile_x_pad = lambda df: df.apply(lambda x: max(0, x["tile_x_max"] - x["image_width"]), axis = "columns"))
                    .assign(tile_y_pad = lambda df: df.apply(lambda x: max(0, x["tile_y_max"] - x["image_height"]), axis = "columns"))
                )
                #display(index_df.merge(spatial_df, how='inner', left_index=True, right_index=True))
                #display(tiled_df)

                tile_metadata = list() 
                max_nodata =  int(0.25*tile_size[0]*tile_size[1]*4) # If there are over this many NODATA(=0s) in a tile, it is skipped 

                if not to_jpg:
                    raster_kwargs = {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'height': tile_size[0], 'width': tile_size[1]}
                    if downsampling_factor is not None:
                        raster_kwargs["height"] = raster_kwargs["height"]//downsampling_factor
                        raster_kwargs["width"] = raster_kwargs["width"]//downsampling_factor

                pbar = tqdm(total = len(tiled_df), desc = "transforming to imagefolder...")
                for _, scene_row in index_df.iterrows():
                    with rio.open(staging_path/scene_row["image_path"], 'r') as raster:
                        image_height, image_width = raster.height, raster.width
                        crs, transform = raster.crs, raster.transform 
                    
                        try:
                            geometry_gdf = gpd.read_file(staging_path/scene_row["label_path"]).to_crs(crs)
                        except Exception as e:
                            print(e)
                            pbar.update(len(tiled_df.loc[[scene_row["image_path"]]]))
                            continue

                        mask: NDArray = features.rasterize(
                            shapes = [(shapely.geometry.shape(g), 255) for g in geometry_gdf.geometry if g is not None], 
                            out_shape = (image_height, image_width), 
                            transform = transform, 
                            dtype = np.uint8
                        )
                
                        for _, row in tiled_df.loc[[scene_row["image_path"]]].iterrows():
                            pbar.update(1)

                            tile_name = f"{scene_row["image_path"].split('/')[-1].removesuffix(".tif")}_{row["tile_x_min"]}_{row["tile_y_min"]}.tif"
                            if to_jpg:
                                tile_name = tile_name.replace(".tif", ".jpg")

                            x_min, x_max = row["tile_x_min"], min(row["tile_x_max"], row["image_width"])
                            y_min, y_max = row["tile_y_min"], min(row["tile_y_max"], row["image_height"])

                            # read image tile from raster
                            # skip particular tile if over :max_nodata of the tile is filled with 0s 
                            # skip entire raster if IOErrors are caught 
                            try:
                                image_tile = raster.read(window=rio.windows.Window.from_slices((y_min, y_max), (x_min, x_max))).transpose(1,2,0)
                                if np.where(image_tile == 0, 1, 0).sum() > max_nodata:
                                    #print(f"skipping {tile_name} for exceding max_nodata")
                                    continue
                            except IOError as e:
                                print(tile_name, e)
                                break
                            
                            tile_metadata.append({"image_path": tile_name} | row.to_dict())
                            #tile_names.append(tile_name)
                            #tile_idxs.append(idx)

                            mask_tile = mask[y_min:y_max, x_min:x_max]
                            if row["tile_x_pad"] != 0 or row["tile_y_pad"] != 0:
                                image_tile = np.pad(image_tile, ((0, row["tile_y_pad"]), (0, row["tile_x_pad"]), (0, 0)))
                                mask_tile = np.pad(mask_tile, ((0, row["tile_y_pad"]), (0, row["tile_x_pad"])))

                            # print(image_tile.shape, mask_tile.shape)

                            # _, axes = plt.subplots(2, 2, figsize = (10 ,10))
                            # axes[0, 0].imshow(image_tile[:,:,:3])
                            # axes[0, 1].imshow(mask_tile, cmap = "gray")

                            if downsampling_factor is not None:
                                image_tile = (rescale(image_tile, (1/downsampling_factor, 1/downsampling_factor, 1)) * 255).astype(np.uint8)
                                mask_tile = (rescale(mask_tile, 1/downsampling_factor).astype(np.uint8) * 255).astype(np.uint8)
                                # axes[1, 0].imshow(image_tile[:,:,:3])
                                # axes[1, 1].imshow(mask_tile, cmap = "gray")
                            

                            if to_jpg:
                                # print(f"writing {images_dir/tile_name}")
                                iio.imwrite(images_dir/tile_name, image_tile[:,:,:3], extension=".jpg", quality = jpg_quality)
                                iio.imwrite(masks_dir/tile_name, mask_tile, extension=".jpg", quality = jpg_quality)
                            
                            else:
                                # print(f"writing {images_dir/tile_name}")
                                tile_kwargs = raster_kwargs | {
                                    "crs": crs, 
                                    "transform": rio.Affine(row["x_res"], 0, row["x_off"], 0, row["y_res"], row["y_off"])
                                }
                                with rio.open(images_dir/tile_name, mode = 'w', **(tile_kwargs|{"count": 4})) as image_raster:
                                    image_raster.write(image_tile.transpose(2,0,1))
                                with rio.open(masks_dir/tile_name, mode = 'w', **(tile_kwargs|{"count": 1})) as mask_raster:
                                    mask_raster.write(mask_tile, 1)

                pbar.close()

                df = (
                    pd.DataFrame(tile_metadata)
                    .assign(image_height = tile_size[0] // (downsampling_factor if downsampling_factor is not None else 1))
                    .assign(image_width = tile_size[1]  // (downsampling_factor if downsampling_factor is not None else 1))
                    .assign(x_res = lambda df: df["x_res"]*(downsampling_factor if downsampling_factor is not None else 1))
                    .assign(y_res = lambda df: df["x_res"]*(downsampling_factor if downsampling_factor is not None else 1))
                )

                index_df = df[["image_path", "subset"]]
                spatial_df = df[spatial_df.columns.to_list()]

                index_df.to_hdf(imagefolder_path/subset/"metadata.h5", key="index", mode="w", format="fixed", complib="zlib", complevel=9)
                spatial_df.to_hdf(imagefolder_path/subset/"metadata.h5", key="spatial", mode="a", format="fixed", complib="zlib", complevel=9)

                return df

            else:
                raise NotImplementedError("scenes are too large to be used without tiling")

        elif to == "hdf5":
            hdf5_path = fs.get_new_dir(cls.local/"hdf5")/"opencities.h5"


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

        tile = scene[row["tile_tl_0"] : min(row["tile_br_0"], row["image_height"]), row["tile_tl_1"] : min(row["tile_br_1"], row["image_width"])]
        tile_size = (int(row["tile_br_0"] - row["tile_tl_0"]), int(row["tile_br_1"] - row["tile_tl_1"]), 4)
        if scene.ndim == 2:
            tile_size = tile_size[:2] 

        if tile.shape != tile_size: 
            pad_width = [_pad(tile_size[0] - tile.shape[0]), _pad(tile_size[1] - tile.shape[1]), (0, 0)]
            tile = np.pad(array = tile, pad_width = pad_width[:2] if tile.ndim == 2 else pad_width)
        return tile