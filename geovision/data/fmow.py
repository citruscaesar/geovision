from typing import Literal, Optional

import os
import cv2
import PIL
import h5py
import json
import torch
import shutil
import shapely
import zipfile
import tarfile
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import imageio.v3 as iio
import pandera.pandas as pa
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from queue import Queue
from skimage.transform import resize
from matplotlib.patches import Rectangle
from torchvision.transforms import v2 as T
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

from geovision.data import Dataset, DatasetConfig
from geovision.io.local import FileSystemIO as fs
from geovision.io.remote import HTTPIO

# FMoW Datasets:
# 1. RGB Multiclass / Multilabel Classification and Detection
#   -> images from large scene classes can be cropped, if needed to preserve context
#   -> dataset must contain tables to interpret the images with different labels 
# 2. MS " 
# 3. Sen2 (RGB / MS) -> Worldview (RGB / MS) Super-Resolution

class FMoW:
    local = Path.home() / "datasets" / "fmow"

    # fmt: off
    class_names = (
        'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 
        'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution',
        'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course',
        'ground_transportation_station', 'helipad', 'hospital', 'impoverished_settlement', 'interchange', 'lake_or_pond', 'lighthouse', 
        'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 
        'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility',
        'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 
        'storage_tank', 'surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility',
        'wind_farm', 'zoo'
    )

    large_image_classes = (
        "airport", "amusement_park", "impoverished_settlement", 
        "nuclear_powerplant", "port", "runway", "shipyard", "space_facility"
    )
    # fmt: on
    crs = "EPSG:4326"
    spatial_ref = "GCS_WGS_1984"
    num_classes = len(class_names)
    means = () 
    std_devs = ()

    default_config = DatasetConfig(
        random_seed = 42,
        tabular_sampler_name = "stratified",
        tabular_sampler_params = dict(
            val_frac = 0.1,
            test_frac = 0.1,
        ),
        image_pre = T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale = True)
        ]), 
        target_pre = None, 
        train_aug = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(0.5),
        ]), 
        eval_aug = T.RandomCrop(512, pad_if_needed = True), 
    )

    index_df_schema = pa.DataFrameSchema(
        columns={
            "image_path": pa.Column(str, coerce=True),
            "label_str": pa.Column(tuple, coerce=True),
        },
        index=pa.Index(int)
    )

    @classmethod
    def extract(cls, subset: Literal["rgb", "ms", "sen2"] = "rgb"):
        """
        downloads fmow-{:dataset} from s3://spacenet-datasets/HostedDatasets/fmow/fmow-{:dataset} to :local_staging/:dataset
        cp train val test and seq directories, ignoring all .json files
        cp groundtruth.tar.bz2 and extract to :local_staging/:dataset/groundtruth/, removing _gt from test and seq dirs, mapping jsons
        """

        assert subset in ("rgb", "ms", "sen2")

        assert shutil.which('s5cmd') is not None, "couldn't find s5cmd on the system"
        if os.environ.get('S3_ENDPOINT_URL') is not None:
            del os.environ['S3_ENDPOINT_URL']
        
        if subset == "sen2":
            for filename in ("fmow-sentinel.tar.gz", "test-gt.csv", "train.csv", "val.csv"):
                HTTPIO.download_url(f"https://stacks.stanford.edu/file/druid:vg497cb6002/{filename}", fs.get_new_dir(cls.local,"staging", subset))
        else:
            subprocess.run([
                "s5cmd", "--no-sign-request", "sync", "--exclude", "*_msrgb.jpg", "--exclude", "*rgb.json", 
                f"s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-{"full" if subset == "ms" else "rgb"}/*", 
                str(fs.get_new_dir(cls.local, "staging", subset))
            ])

    @classmethod
    def download(cls, subset: Literal["rgb_clf", "ms_clf", "sen2_rgb_super", "sen2_ms_super"], s3_endpoint_url: Optional[str] = "https://sin1.contabostorage.com"):
        """
        downloads and extracts train.zip, val.zip, test.zip, seq.zip, groundtruth.tar.bz2 to :local_staging/rgb/
        """
        assert subset in ("rgb_clf", "sen2_super_res")
        assert shutil.which('s5cmd') is not None, "couldn't find s5cmd on the system"
        if os.environ.get('S3_ENDPOINT_URL') is None and s3_endpoint_url is not None:
            os.environ['S3_ENDPOINT_URL'] = s3_endpoint_url 
        subprocess.run(["s5cmd", "cp", "--sp", f"s3://fmow/hdf5/{subset}.h5", str(fs.get_new_dir(cls.local, "hdf5"))])

    @classmethod
    def load(
        cls, 
        table: Literal["index", "spatial", "spectral", "temporal", "radiometric"], 
        src: Literal["staging", "imagefolder", "hdf5"],
        subset: Literal["fmow_rgb_clf", "fmow_sen2_super_res"]
    ) -> pd.DataFrame:

        assert src in ("staging", "imagefolder", "hdf5")
        assert subset in ("fmow_rgb_clf", "fmow_sen2_super_res") 
        
        if src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"{subset}.h5", key = table, mode = 'r')

        def _get_metadata_columns() -> list:
            columns = [
                "parent_dir", "label_str", "bbox", "img_filename", "gsd", "img_width", "img_height", "mean_pixel_height", "mean_pixel_width",
                "utm", "country_code", "cloud_cover", "timestamp", "scan_direction", "approximate_wavelengths", "catalog_id", 
                "sensor_platform_name", "raw_location", "epsg", "spatial_reference", # "abs_cal_factors", NOTE: is missing in some .json files
            ]
            columns += [f"{x}{y}_dbl" for x in ("pan_resolution", "multi_resolution", "target_azimuth", "off_nadir_angle") for y in ("", "_start", "_end", "_min", "_max")]
            columns += [f"{x}{y}_dbl" for x in ("sun_azimuth", "sun_elevation") for y in ("", "_min", "_max")]
            return columns

        def _get_filename(parent_dir: str, img_filename: str, rename: dict):
            if parent_dir in rename.keys():
                parent_dir = rename[parent_dir]
                img_filename = '_'.join([parent_dir.split('/')[-1]] + img_filename.split('_')[-2:])
            return Path(parent_dir, img_filename).as_posix()

        if subset == "fmow_rgb_clf":
            assert src in ("archive", "imagefolder")

            def _process_metadata(metadata: dict, rename: dict) -> dict[str, pd.DataFrame]:
                df = (
                    pd.DataFrame(metadata)
                    .loc[lambda df: df["label_str"] != "false_detection"]
                    .assign(image_path = lambda df: df.apply(lambda x: _get_filename(x["parent_dir"], x["img_filename"], rename), axis = 1))
                    .assign(split = lambda df: df.apply(lambda x: str(x["image_path"]).split('/')[0], axis = 1))
                    .sort_values(["split", "parent_dir", "timestamp"])
                    .assign(label_bbox = lambda df: df["bbox"].apply(lambda x: cls.get_corners_from_bbox(x)))
                    .drop(columns = ["split", "parent_dir", "img_filename", "bbox"])
                    .groupby("image_path")
                    .agg({k: "first" for k in columns if k not in ("parent_dir", "bbox", "img_filename", "image_path")} | {"label_str": tuple, "label_bbox": tuple})
                    .reset_index(drop = False)
                )
                columns.remove("parent_dir")
                columns.remove("bbox")
                columns.remove("img_filename")

                index_cols = ["image_path", "label_str", "label_bbox", "catalog_id"]
                index_df = df[index_cols].copy()

                spatial_cols = ["img_width", "img_height", "mean_pixel_height", "mean_pixel_width", "gsd", "utm", "country_code", "raw_location", "epsg", "spatial_reference"]
                spatial_df = df[spatial_cols].copy() 
                spatial_df = spatial_df.rename({"img_width": "image_width", "img_height": "image_height"}, axis = 'columns')
                
                temporal_cols = ["timestamp"]
                temporal_df = df[temporal_cols].copy()
                temporal_df["timestamp"] = pd.to_datetime(temporal_df["timestamp"], format = "mixed")
                
                # # spectral_cols = ["abs_cal_factors", "approximate_wavelengths"]
                # # abs_cal_df = (
                    # # pd.concat(df["abs_cal_factors"].apply(lambda x: pd.DataFrame(x).set_index("band").transpose()).tolist())
                    # # .reset_index(drop = True)
                    # # .rename_axis(None, axis = 1)
                    # # .add_suffix("_abs_cal_factor")
                # # )
                # # spectral_df = pd.concat([df["approximate_wavelengths"], abs_cal_df], axis = 1)  
                spectral_cols = ["approximate_wavelengths"]
                spectral_df = df[["approximate_wavelengths"]]

                radiometric_cols = sorted(set(columns).difference(index_cols + spatial_cols + spectral_cols + temporal_cols))
                radiometric_df = df[radiometric_cols].copy()

                return {"index": index_df, "spatial": spatial_df, "spectral": spectral_df, "temporal": temporal_df, "radiometric": radiometric_df}

            if src == "staging":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "rgb", "groundtruth.tar.bz2")
                try:
                    #raise OSError
                    return pd.read_hdf(archive_path.parent/"metadata.h5", key = table, mode = "r")
                except OSError:
                    assert table in ("index", "spatial", "spectral", "temporal", "radiometric")

                    columns = _get_metadata_columns() 
                    metadata = {k:list() for k in columns}

                    with tarfile.open(archive_path, mode = "r:bz2") as tf:
                        for tar_json in tqdm([x for x in tf.getnames() if f"_{"msrgb" if subset == "ms" else "rgb"}.json" in x]):
                            parsed_json = json.load(tf.extractfile(tar_json))
                            for bbox in parsed_json["bounding_boxes"][1:]:
                                metadata["parent_dir"].append('/'.join(str(tar_json).split('/')[-4:-1]))
                                metadata["label_str"].append(bbox["category"])
                                metadata["bbox"].append(bbox["box"])
                                for key in columns[3:]:
                                    metadata[key].append(parsed_json[key])

                        test_df = (
                            pd.read_json(tf.extractfile("test_gt_mapping.json"))
                            .assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
                        )
                        seq_df = (
                            pd.read_json(tf.extractfile("seq_gt_mapping.json"))
                            .assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
                        )
                        rename = dict(zip(test_df["input"], test_df["output"])) | dict(zip(seq_df["input"], seq_df["output"]))
                    
                    dfs = _process_metadata(metadata, rename)
                    dfs.update({"test_index": test_df, "seq_index": seq_df})
                    for key in dfs:
                        dfs[key].to_hdf(archive_path.parent/"metadata.h5", key = key, mode = 'a', format = 'fixed', complevel=9, complib='zlib')
                    return dfs[table]
                    
            elif src == "imagefolder":
                imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder", "rgb")
                try:
                    return pd.read_hdf(imagefolder_path/"metadata.h5", key = table, mode = "r")
                except OSError:
                    assert table in ("index", "spatial", "spectral", "temporal", "radiometric")

                    columns = _get_metadata_columns() 
                    metadata = {k:list() for k in columns}

                    groundtruth_path = imagefolder_path/"groundtruth"
                    for json_path in tqdm(list(groundtruth_path.rglob(f"*_{"msrgb" if subset == "fmow_ms" else "rgb"}.json"))):
                        with open(json_path) as f:
                            parsed_json = json.load(f)
                        for bbox in parsed_json["bounding_boxes"][1:]:
                            metadata["parent_dir"].append('/'.join(str(json_path).split('/')[-4:-1]))
                            metadata["label_str"].append(bbox["category"])
                            metadata["bbox"].append(bbox["box"])
                            for key in columns[3:]:
                                metadata[key].append(parsed_json[key])

                    test_df = (
                        pd.read_json(groundtruth_path/"test_gt_mapping.json")
                        .assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
                    )
                    seq_df = (
                        pd.read_json(groundtruth_path/"seq_gt_mapping.json")
                        .assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
                    )
                    rename = dict(zip(test_df["input"], test_df["output"])) | dict(zip(seq_df["input"], seq_df["output"]))

                    dfs = _process_metadata(metadata, rename)
                    dfs.update({"test_index": test_df, "seq_index": seq_df})
                    for key in dfs:
                        dfs[key].to_hdf(imagefolder_path/"metadata.h5", key = key, mode = 'a', format = 'fixed', complevel=9, complib='zlib')
                    return dfs[table]

        elif subset == "sen2":
            def _get_image_path(row: pd.Series, split: str):
                return f"{split}/{row["category"]}/{row["category"]}_{row["location_id"]}/{row["category"]}_{row["location_id"]}_{row["image_id"]}.tif"

            if src == "archive":
                archive_path = fs.get_valid_file_err(cls.local, "sentinel", "archives", "groundtruth.zip")
                try:
                    return pd.read_hdf(archive_path.parent/"metadata.h5", key = table, mode = 'r')
                except OSError:
                    split_dfs = list()
                    with zipfile.ZipFile(archive_path, mode = 'r') as zf:
                        for split in ("train", "val", "test"):
                            split_dfs.append(
                                pd.read_csv(zf.read(f"{split}.csv"), index_col=0)
                                .assign(image_path = lambda df: df.apply(lambda x: _get_image_path(x, "train"), axis = 1))
                            )
                    df = pd.concat([split_dfs], axis = 0).drop(colums = ["categeory", "location_id", "image_id"])
                    df.to_hdf(imagefolder_path/"metadata.h5", key = table, mode = 'a', format = 'fixed', complevel=9, complib='zlib')
                    return df

            elif src == "imagefolder":
                imagefolder_path = fs.get_valid_dir_err(cls.local, "sentinel")
                try: 
                    return pd.read_hdf(imagefolder_path/"metadata.h5", key = table, mode = 'r')
                except OSError:
                    assert table == "index"
                    split_dfs = list()
                    for split in ("train", "val", "test"):
                        split_dfs.append(
                            pd.read_csv(imagefolder_path/"groundtruth"/f"{split}.csv", index_col=0)
                            .assign(image_path = lambda df: df.apply(lambda x: _get_image_path(x, split), axis = 1))
                        )
                    df = pd.concat([split_dfs], axis = 0).drop(columns = ["category", "location_id", "image_id"])
                    df.to_hdf(imagefolder_path/"metadata.h5", key = table, mode = 'a')
                    return df

    @classmethod
    def transform(
        cls,
        src: Literal["staging", "imagefolder"],
        to: Literal["imagefolder", "hdf5"],
        subset: Literal["fmow_rgb_clf", "fmow_sen2_super_res"],
        crop_to_bbox: bool = False,
        resize_to: Optional[int] = None,
        jpeg_quality: int = 90, # jpeg_quality = 0 means no encode to jpeg, then resize to must not be None
        num_parts: int = 10,
        num_proc: int = 3,
        use_optimized: bool = True  # Use optimized writer with cv2 and threading
    ):
        assert src == "imagefolder"
        assert to == "hdf5"
        assert subset in ("fmow_rgb_clf", "fmow_sen2_super_res")

        assert isinstance(jpeg_quality, int) and (jpeg_quality <= 100) and (jpeg_quality >= 0)

        if jpeg_quality == 0:
            assert isinstance(resize_to, int), "resize_to must be provided if encoding to hdf5 and jpeg conversion is off"
       
        if subset == "fmow_rgb_clf":
            # Assert the source files exist, as archive or imagefolder
            imagefolder_path = fs.get_valid_dir_err(FMoW.local / "imagefolder" / "rgb")

            # update image_paths for seq and test splits
            index_df = cls.load('index', 'imagefolder', subset)
            index_df["image_path"] = index_df["image_path"].apply(lambda x: x.replace("seq_gt", "seq").replace("test_gt", "test"))

            seq_index_df = cls.load('seq_index', 'imagefolder', subset)
            test_index_df = cls.load('test_index', 'imagefolder', subset)
            seq_rename = {f"{x}/{x.split('/')[-1]}"  : f"{y}/{y.split('/')[-1]}" for (x, y) in zip(seq_index_df["input"], seq_index_df["output"])}
            test_rename = {f"{x}/{x.split('/')[-1]}"  : f"{y}/{y.split('/')[-1]}" for (x, y) in zip(test_index_df["input"], test_index_df["output"])}
            rename = seq_rename | test_rename

            seq_test_df = index_df[index_df["image_path"].str.split('/').str[0].isin(("seq", "test"))].copy()
            seq_test_df["_prefix"] = seq_test_df["image_path"].str.split('_').str[:-2].str.join('_')
            seq_test_df["corr_image_path"] = seq_test_df.apply(lambda x: x["image_path"].replace(x["_prefix"], rename[x["_prefix"]]), axis = 1)
            seq_test_df = seq_test_df.drop(columns = "_prefix")

            index_df = index_df.drop(index = seq_test_df.index)
            index_df["corr_image_path"] = index_df["image_path"]
            index_df = pd.concat([index_df, seq_test_df]).sort_index()

            # Check :crop_to_bbox is true, load the spatial dataframe, take the union of all bboxes per image, apply square crop and update spatial geometry.
            spatial_df = cls.load('spatial', 'imagefolder', subset)

            if crop_to_bbox:
                print("crop_to_bbox")
                df = index_df.merge(spatial_df[["img_height", "img_width"]], how="left", left_index=True, right_index=True) 
                df["union_bbox"] = df["label_bbox"].apply(cls._get_bbox_union)
                df["union_bbox"] = df.apply(cls._get_bbox_square, axis = 1)
                df = df.drop(columns = ["img_height", "img_width"])

                # Translate label_bbox in the image space
                print("translate_bbox")
                translated_bboxes = list()
                for _, row in df.iterrows():
                    origin = row["union_bbox"][0], row["union_bbox"][1]
                    label_bboxes = list()
                    for label_bbox in row["label_bbox"]:
                        label_bboxes.append(cls._translate_bbox(origin, label_bbox))
                    translated_bboxes.append(tuple(label_bboxes))
                df["label_bbox"] = translated_bboxes

                # Translate raw_location in the geographic space
                print("translate_polygon")
                spatial_df = spatial_df.merge(right = df[["union_bbox"]], how="left", left_index=True, right_index=True)
                spatial_df["bbox_polygon"] = spatial_df.apply(cls._get_cropped_polygon, axis = 1)
                spatial_df[["img_height", "img_width"]] = spatial_df["union_bbox"].apply(cls._get_bbox_dims).to_list()
                spatial_df = spatial_df.drop(columns = ["union_bbox"])

                index_df = df.copy()
            
            # # Check if :resize_to is not None, resize and pad all images to :(int)resize_to. 
            # # This renders geometry meaningless, and is purely for fast training, so remove the irrelevant columns.
            # if resize_to is not None:
                # ...
            
            # print("check if all images exist on disk")
            # for idx, row in tqdm(index_df.iterrows(), total=len(index_df)):
                # assert (imagefolder_path / row["corr_image_path"]).is_file(), f"dne, {idx} :: {row["corr_image_path"]}"

            print(f"writing to h5 as {num_parts} parts across {num_proc} processes ...")
            hdf5_path = fs.get_new_dir(FMoW.local / "hdf5") / f"{subset}.h5"

            args = list()
            for i, idxs in enumerate(np.array_split(index_df.index, num_parts)):
                args.append((index_df.iloc[idxs], crop_to_bbox, resize_to, jpeg_quality, hdf5_path.parent/f"{hdf5_path.stem}_part={i}.h5", imagefolder_path))

            num_proc = num_proc or cpu_count() - 1
            writer_func = cls._write_to_hdf_optimized if use_optimized else cls._write_to_hdf
            with Pool(num_proc) as pool:
                pool.starmap(writer_func, args)

            with h5py.File(hdf5_path, mode='w') as f:
                vlen_dtype = h5py.special_dtype(vlen = np.uint8)
                layout = h5py.VirtualLayout(shape = len(index_df), dtype = vlen_dtype)
                for idx, args_tuple in enumerate(args):
                    vds_df, vds_path = args_tuple[0], args_tuple[-2]
                    layout[idx*len(vds_df): (idx+1)*len(vds_df)] = h5py.VirtualSource(vds_path, "images", len(index_df), vlen_dtype)
                f.create_virtual_dataset("images", layout)

            # cls._write_to_hdf(index_df, crop_to_bbox, resize_to, jpeg_quality, hdf5_path, imagefolder_path)
                            
            index_df["label_idx"] = index_df["label_str"].apply(lambda x: tuple(FMoW.class_names.index(y) for y in x))
            index_df = index_df[["image_path", "label_str", "label_idx", "label_bbox", "catalog_id"]]

            index_df.to_hdf(hdf5_path, key="index", mode='r+')
            spatial_df.to_hdf(hdf5_path, key="spatial", mode='r+')

            FMoW.load('temporal', 'imagefolder', subset).to_hdf(hdf5_path, key="temporal", mode="r+")
            FMoW.load('radiometric', 'imagefolder', subset).to_hdf(hdf5_path, key="radiometric", mode="r+")
            FMoW.load('spectral', 'imagefolder', subset).to_hdf(hdf5_path, key="spectral", mode="r+")

        # elif subset == "sen2_super_res":
            # # Assert the source files exist, as archive or imagefolder
            # # Load the dataframes from the source directories metadata
            # # Create the backwards rename dict using test_index and seq_index dataframes from the metadata
            # # Assert crop_to_bbox is False and resize_to is None.
            # # Create a injective mapping between rgb and sen2_ms, by filtering samples by time and category
            # # ... understand the geometry mapping b/w the two
            # pass

    @staticmethod
    def _write_to_hdf(df: pd.DataFrame, crop_to_bbox: True, resize_to: int, jpeg_quality: int, hdf5_path: Path, imagefolder_path: Path):
        with h5py.File(hdf5_path, 'w') as f:
            if jpeg_quality > 0:
                images = f.create_dataset("images", shape=len(df), dtype=h5py.special_dtype(vlen=np.uint8))
            else:
                images = f.create_dataset("images", shape=(len(df), resize_to, resize_to, 3), dtype=np.uint8)

            PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
            for idx, row in tqdm(df.iterrows(), total = len(df), desc=f"writing to {hdf5_path.stem}"):
                image = iio.imread(imagefolder_path / row["corr_image_path"], extension=".jpg")
                if crop_to_bbox:
                    image = image[row["union_bbox"][0]:row["union_bbox"][2], row["union_bbox"][1]:row["union_bbox"][3]]
                if resize_to is not None:
                    image = resize(image, (resize_to, resize_to), preserve_range=True, anti_aliasing=True).astype(np.uint8)
                if jpeg_quality > 0:
                    image = np.frombuffer(iio.imwrite("<bytes>", image, extension=".jpg", quality=jpeg_quality), dtype=np.uint8)
                images[idx] = image
        df.to_hdf(hdf5_path, key="index", mode='r+')

    @staticmethod
    def _write_to_hdf_optimized(df: pd.DataFrame, crop_to_bbox: bool, resize_to: Optional[int], jpeg_quality: int, hdf5_path: Path, imagefolder_path: Path, prefetch_size: int = 6):
        """
        Optimized HDF5 writer with I/O prefetching and fast cv2 operations.

        Uses threading to overlap I/O (reading images) with CPU work (processing/encoding),
        and cv2 for 2-3× faster JPEG decoding, resizing, and encoding compared to imageio/skimage.

        Args:
            df: DataFrame with image metadata (must contain 'corr_image_path' column)
            crop_to_bbox: Whether to crop images to their bounding boxes (requires 'union_bbox' column)
            resize_to: Target size for resizing (None to skip resizing)
            jpeg_quality: JPEG quality 1-100 (0 to store raw arrays instead of JPEG)
            hdf5_path: Output HDF5 file path
            imagefolder_path: Root directory containing images
            prefetch_size: Number of images to prefetch (default 6, optimal for SSDs)
        """
        # Queue for prefetched images (size=6 optimal for SSDs with small images)
        image_queue = Queue(maxsize=prefetch_size)
        error_queue = Queue()  # For error handling from the reader thread

        def read_images_threaded():
            """Background thread: continuously read images ahead of processing."""
            try:
                for idx, row in df.iterrows():
                    image_path = str(imagefolder_path / row["corr_image_path"])
                    # cv2.imread releases GIL during I/O and is 2-3× faster than imageio
                    image = cv2.imread(image_path)

                    if image is None:
                        error_queue.put(f"Failed to read image: {image_path}")
                        # Send None to maintain queue ordering
                        image_queue.put((idx, row, None))
                    else:
                        # Convert BGR (cv2 default) to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_queue.put((idx, row, image))

                # Signal completion
                image_queue.put(None)
            except Exception as e:
                error_queue.put(f"Reader thread error: {str(e)}")
                image_queue.put(None)

        # Start background reader thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            reader_future = executor.submit(read_images_threaded)

            # Create HDF5 file and dataset
            with h5py.File(hdf5_path, 'w') as f:
                if jpeg_quality > 0:
                    images = f.create_dataset("images", shape=len(df),
                                            dtype=h5py.special_dtype(vlen=np.uint8))
                else:
                    images = f.create_dataset("images", shape=(len(df), resize_to, resize_to, 3),
                                            dtype=np.uint8)

                PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
                pbar = tqdm(total=len(df), desc=f"writing to {hdf5_path.stem}")

                processed_count = 0
                while True:
                    # Check for reader thread errors
                    if not error_queue.empty():
                        error_msg = error_queue.get()
                        print(f"\nWarning: {error_msg}")

                    # Get next image from queue (blocks if queue is empty)
                    item = image_queue.get()

                    if item is None:
                        # Sentinel value - processing complete
                        break

                    idx, row, image = item

                    if image is None:
                        # Skip failed images but maintain indexing
                        pbar.update(1)
                        continue

                    # Crop if needed
                    if crop_to_bbox and "union_bbox" in row:
                        y_min, x_min, y_max, x_max = row["union_bbox"]
                        image = image[y_min:y_max, x_min:x_max]

                    # Resize if needed (cv2 is 5-10× faster than skimage)
                    if resize_to is not None:
                        # INTER_AREA is best for downsampling (preserves image quality via pixel area averaging)
                        image = cv2.resize(image, (resize_to, resize_to),
                                         interpolation=cv2.INTER_AREA)

                    # Encode to JPEG if needed
                    if jpeg_quality > 0:
                        # cv2.imencode is 2-3× faster than imageio
                        # Convert back to BGR for JPEG encoding
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        success, encoded = cv2.imencode('.jpg', image_bgr,
                                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        if success:
                            image = np.frombuffer(encoded, dtype=np.uint8)
                        else:
                            print(f"\nWarning: Failed to encode image at index {idx}")
                            pbar.update(1)
                            continue

                    # Write to HDF5
                    images[processed_count] = image
                    processed_count += 1
                    pbar.update(1)

                pbar.close()

        # Save dataframe metadata
        df.to_hdf(hdf5_path, key="index", mode='r+')

    @staticmethod
    def get_rectangle_from_bbox(tl: tuple[int, int], br: tuple[int, int], **kwargs) -> Rectangle:
            #          bbox                       rect 
            #         (tl, br)                ((x,y), w, h)
            #   (y,x) ------------|        |----------------|
            #    |                |        |                |
            #    |                |        h                |
            #    |                |   ->   |                |
            #    |                |        |                |
            #    |--------------(y,x)      (x,y) -----w-----|
            return Rectangle((tl[1], br[0]), br[1] - tl[1], tl[0] - br[0], **kwargs)

    @staticmethod
    def _get_bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _translate_bbox(origin: tuple[int, int], bbox: tuple[int, int, int, int]):
        # origin: (y_min, x_min), bbox: (y_min, x_min, y_max, x_max)
        return bbox[0] - origin[0], bbox[1] - origin[1], bbox[2], bbox[3]
   
    @staticmethod
    def _get_bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        y_min, x_min, y_max, x_max = int(9e4), int(9e4), 0, 0 
        for bbox in bboxes:
            if bbox[0] < y_min: 
                y_min = bbox[0]
            if bbox[1] < x_min: 
                x_min = bbox[1]
            if bbox[2] > y_max: 
                y_max = bbox[2]
            if bbox[3] > x_max: 
                x_max = bbox[3]
        return y_min, x_min, y_max, x_max

    @staticmethod
    def _get_bbox_square(row: pd.Series) -> pd.Series:
        # by row-major convention - dim_0: height, dim_1: width
        image_dims = row["img_height"], row["img_width"]
        bbox_tl, bbox_br = (row["union_bbox"][0], row["union_bbox"][1]), (row["union_bbox"][2], row["union_bbox"][3])
        bbox_dims = bbox_br[0] - bbox_tl[0], bbox_br[1] - bbox_tl[1]

        # bbox will be modified along the smaller dimension of the bbox, sdim
        sdim = np.argmin(bbox_dims)
        ldim = np.abs(sdim - 1)

        # skip over images smaller than 2000x2000
        # if image_dims[ldim] < 2000:
            # return 0, 0, *image_dims
        
        # calculate corrections along the smaller dimension 
        avail_before = bbox_tl[sdim] 
        avail_after = image_dims[ldim] - bbox_br[ldim]
        reqd = bbox_dims[ldim] - bbox_dims[sdim]
        if reqd % 2 == 0:
            reqd_before, reqd_after = reqd // 2, reqd // 2 
        else:
            reqd_before, reqd_after = (reqd // 2) + 1, reqd // 2

        # ajust corrections if bbox is not roughly in the center and there aren't enough pixels along
        # the smaller dimension; as much of the image as possible will be used
        if avail_before < reqd_before:
            reqd_after = min(avail_after, reqd_after + reqd_before - avail_before)
            reqd_before = avail_before
        elif avail_after < reqd_after:
            reqd_before = min(avail_before, reqd_before + reqd_after - avail_after)
            reqd_after = avail_after
        
        # apply the corrections
        new_bbox_tl = list(bbox_tl)
        new_bbox_tl[sdim] -= reqd_before
        new_bbox_br = list(bbox_br)
        new_bbox_br[sdim] += reqd_after

        return *new_bbox_tl, *new_bbox_br

    # NOTE: Generated by Claude Code 
    @staticmethod
    def _get_cropped_polygon(row: pd.Series) -> shapely.geometry.Polygon:
        """
        Compute the geographic polygon (lat/lon) for a cropped image region using bilinear interpolation.
        
        This function correctly handles images that are rotated or skewed in geographic space by treating
        the original image polygon as a quadrilateral and using bilinear interpolation to map pixel 
        coordinates to geographic coordinates.
        
        The function automatically detects the corner ordering from the polygon coordinates.
        
        Args:
            row: DataFrame row containing:
                - raw_location: original image polygon (lon, lat format) - a quadrilateral
                - union_bbox: crop bounds (y_min, x_min, y_max, x_max) in pixel coordinates
                - img_width: original image width in pixels
                - img_height: original image height in pixels
        
        Returns:
            shapely.Polygon representing the geographic extent of the cropped region
        """
        # Extract original polygon corners (in lon, lat format)
        if isinstance(row["raw_location"], str):
            original_polygon = shapely.wkt.loads(row["raw_location"])
        else:
            original_polygon = row["raw_location"]

        coords = list(original_polygon.exterior.coords[:4])
        # Identify corners by their geographic position (min/max lon/lat)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Find which coordinate corresponds to which corner
        # We need to map: NW (top-left in image), NE (top-right), SE (bottom-right), SW (bottom-left)
        # Image coordinates: top = min_y (row 0), bottom = max_y, left = min_x (col 0), right = max_x
        # Geographic: North = max_lat, South = min_lat, West = min_lon, East = max_lon
        
        P_corners = {}
        for lon, lat in coords:
            # Classify as North/South and West/East
            is_north = lat > (min_lat + max_lat) / 2
            is_west = lon < (min_lon + max_lon) / 2
            
            if is_north and is_west:
                P_corners['NW'] = np.array([lon, lat])  # Top-left in image
            elif is_north and not is_west:
                P_corners['NE'] = np.array([lon, lat])  # Top-right in image
            elif not is_north and not is_west:
                P_corners['SE'] = np.array([lon, lat])  # Bottom-right in image
            elif not is_north and is_west:
                P_corners['SW'] = np.array([lon, lat])  # Bottom-left in image
        
        # Map to image coordinate system
        # Top-left (0,0) in image = North-West in geography
        # Top-right (0,width) in image = North-East in geography
        # Bottom-right (height,width) in image = South-East in geography
        # Bottom-left (height,0) in image = South-West in geography
        P_tl = P_corners['NW']  # Top-left: y=0, x=0
        P_tr = P_corners['NE']  # Top-right: y=0, x=width
        P_br = P_corners['SE']  # Bottom-right: y=height, x=width
        P_bl = P_corners['SW']  # Bottom-left: y=height, x=0
        
        # Extract crop bounds in pixel space: (y_min, x_min, y_max, x_max)
        y_min, x_min, y_max, x_max = row["union_bbox"]
        img_height, img_width = row["img_height"], row["img_width"]
        
        # Normalize pixel coordinates to [0, 1] range
        # u = x / width (horizontal: 0=left/west, 1=right/east)
        # v = y / height (vertical: 0=top/north, 1=bottom/south)
        u_min = x_min / img_width
        u_max = x_max / img_width
        v_min = y_min / img_height
        v_max = y_max / img_height
        
        # Bilinear interpolation function
        def bilinear_interp(u, v):
            """
            Bilinear interpolation on quadrilateral:
            P(u,v) = (1-u)(1-v)*P_tl + u(1-v)*P_tr + uv*P_br + (1-u)v*P_bl
            
            where u goes from 0 (left/west) to 1 (right/east)
            and v goes from 0 (top/north) to 1 (bottom/south)
            """
            return (1 - u) * (1 - v) * P_tl + \
                u * (1 - v) * P_tr + \
                u * v * P_br + \
                (1 - u) * v * P_bl
        
        # Compute the four corners of the cropped region in geographic space
        new_tl = bilinear_interp(u_min, v_min)  # top-left of crop
        new_tr = bilinear_interp(u_max, v_min)  # top-right of crop
        new_br = bilinear_interp(u_max, v_max)  # bottom-right of crop
        new_bl = bilinear_interp(u_min, v_max)  # bottom-left of crop
        
        # Create and return the new polygon (close it by adding the first point at the end)
        return shapely.geometry.Polygon([new_tl, new_tr, new_br, new_bl, new_tl])

        
class FMoW_RGB_HDF5_Classification(Dataset):
    name = "fmow_rgb_clf"
    task = "classification"
    subtask = "multiclass"
    storage = "hdf5"
    class_names = FMoW.class_names 
    num_classes = FMoW.num_classes 
    root = FMoW.local/"hdf5"/"fmow_rgb_clf.h5"
    schema = FMoW.index_df_schema 
    config = FMoW.default_config
    loader = FMoW.load

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=False)

        PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3

        # self._root = fs.get_valid_file_err(FMoWETL.local_rgb, "hdf5", "fmow_rgb_multiclass.h5")
        # self._split = self.get_valid_split_err(split)
        # self._config = config or FMoWETL.default_config
        # self._df = self._config.verify_and_get_df(schema = self.df_schema, fallback_df = pd.read_hdf(self._root, key = "dataset_df", mode = 'r'))
        # self._split_df = self._config.verify_and_get_split_df(df = self._df, schema = self.df_schema, split = self._split)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.df.iloc[idx]

        with h5py.File(self.root, mode="r") as f:
            image = iio.imread(BytesIO(f["images"][idx_row["df_idx"]]), extension=".jpeg")

        image = self._config.image_pre(image)
        if self._split in ("train", "trainvaltest"):
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        
        return image, idx_row["label_idx"][0], idx_row["df_idx"]

# # Transformations
    # @classmethod
    # def transform_to_classification_imagefolder(cls, dataset: Literal["rgb", "ms"] = "rgb", classification: Literal["multiclass", "multilabel"] = "multiclass"):
        # if dataset == "rgb":
            # local, staging = cls.local_rgb, cls.local_staging / "rgb"
        # elif dataset == "ms":
            # local, staging = cls.local_ms, cls.local_staging / "ms"
        # else:
            # raise ValueError(f"invalid :dataset, expected either rgb or ms, got {dataset}")

        # if classification == "multiclass":
            # df = cls.get_multiclass_classification_df_from_metadata()
        # elif classification == "multilabel":
            # df = cls.get_multilabel_classification_df_from_metadata()
        # else:
            # raise ValueError(f"invalid :classification, expected either multiclass or multilabel, got {dataset}")

        # imagefolder = fs.get_new_dir(local, "imagefolder")

        # for subdir in df["image_path"].apply(lambda x: x.parent).unique():
            # fs.get_new_dir(imagefolder, subdir)

        # for _, row in tqdm(df.iterrows(), total = len(df)):
            # if row["img_height"] >= 2000 or row["img_width"] >= 2000:
                # iio.imwrite(
                    # uri = imagefolder/row["image_path"],
                    # image = iio.imread(staging / row["image_path"])[row["outer_bbox_tl_0"]:row["outer_bbox_br_0"], row["outer_bbox_tl_1"]:row["outer_bbox_br_1"]].astype(np.uint8),
                    # extension = ".jpg"
                # )
            # else:
                # shutil.copy(imagefolder/row["image_path"], staging/row["image_path"])

        # df["img_height"] = df.apply(lambda x: x["outer_bbox_br_0"] - x["outer_bbox_tl_0"], axis = 1)
        # df["img_width"] = df.apply(lambda x: x["outer_bbox_br_1"] - x["outer_bbox_tl_1"], axis = 1)
        # df = df[["image_path", "label_str", "img_width", "img_height", "inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]]
        # df.to_hdf(imagefolder/"dataset.h5", key = "dataset_df", mode = "w")

    # @classmethod
    # def transform_to_multiclass_classification_hdf5(cls, dataset: Literal["rgb", "ms"] = "rgb"):
        # PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        # if dataset == "rgb":
            # local, staging = cls.local_rgb, cls.local_staging / "rgb"
        # elif dataset == "ms":
            # local, staging = cls.local_ms, cls.local_staging / "ms"
        # else:
            # raise ValueError(f"invalid :dataset, expected either rgb or ms, got {dataset}")

        # df = cls.get_multiclass_classification_df_from_metadata()
        # ds = fs.get_new_dir(local, "hdf5") / f"fmow_{dataset}_multiclass.h5"

        # with h5py.File(ds, mode = "w") as f:
            # images = f.create_dataset("images", shape = len(df), dtype = h5py.special_dtype(vlen=np.uint8))
            # for idx, row in tqdm(df.iterrows(), total = len(df)):
                # image = iio.imread(staging / row["image_path"]).astype(np.uint8)
                # image = image[row["outer_bbox_tl_0"]:row["outer_bbox_br_0"], row["outer_bbox_tl_1"]:row["outer_bbox_br_1"]]
                # image = iio.imwrite('<bytes>', image, extension=".jpg")
                # images[idx] = np.frombuffer(image, dtype = np.uint8)

        # df["img_height"] = df.apply(lambda x: x["outer_bbox_br_0"] - x["outer_bbox_tl_0"], axis = 1)
        # df["img_width"] = df.apply(lambda x: x["outer_bbox_br_1"] - x["outer_bbox_tl_1"], axis = 1)
        # df = df[["image_path", "label_str", "img_width", "img_height", "inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]]
        # df.to_hdf(ds, key = "index", mode = "r+")
    
    # def transform_to_superresolution_imagefolder(cls):
        # def get_category(image_path: Path):
            # image_path = str(image_path)
            # category = '/'.join(image_path.split('/')[:-1])
            # if category in rename_dict.keys():
                # return rename_dict[category]
            # return category

        # test_df = pd.read_json(cls.local_rgb/"groundtruth"/"test_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
        # seq_df = pd.read_json(cls.local_rgb/"groundtruth"/"seq_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
        # rename_dict = dict(zip(test_df["output"], test_df["input"])) | dict(zip(seq_df["output"], seq_df["input"]))

        # sen_df = cls.get_sentinel_metadata_df()
        # sen_df["timestamp"] = pd.to_datetime(sen_df["timestamp"], format = "mixed")
        # sen_df["category"] = sen_df["image_path"].apply(lambda x: '/'.join(x.split('/')[:-1]))

        # rgb_df = (
            # cls.get_metadata_df()
            # .assign(split = lambda df: df["image_path"].apply(lambda x: str(x).split('/')[0]))
            # .assign(category = lambda df: df["image_path"].apply(lambda x: get_category(x)))
        # )
        # rgb_df = rgb_df[rgb_df["label_str"] != "false_detection"]
        # rgb_df = rgb_df[rgb_df["split"] != "seq"]

        # time_range_df = rgb_df.groupby("category").agg({"timestamp": ["min", "max"]}).reset_index(drop = False)
        # time_range_df.columns = ["category", "timestamp_min", "timestamp_max"]
        # sen_df = pd.merge(sen_df, time_range_df, how="left", on="category")
        # sen_df = sen_df[sen_df.apply(lambda x: x["timestamp"] > x["timestamp_min"] and x["timestamp"] < x["timestamp_max"], axis = 1)]

        # sen_categories = set(sen_df["category"].unique())

        # rgb_df = rgb_df[rgb_df["category"].apply(lambda x: x in sen_categories)]
        # rgb_df = rgb_df[["image_path", "timestamp", "geometry", "category"]].sort_values(by = ["category", "timestamp"], ascending=[True, True]).reset_index(drop = True)
        # sen_df = sen_df.drop(columns = ["timestamp_min", "timestamp_max"]).sort_values(by = ["category", "timestamp"], ascending=[True, True]).reset_index(drop = True)

        # geometry_df = pd.concat([rgb_df, sen_df], axis = 0).groupby("category").agg({"geometry": shapely.intersection_all}).reset_index(drop = False)
        # geometry_df.columns = ["category", "intersection"]

        # rgb_inter_df = pd.merge(rgb_df, geometry_df, "left", "category")
        # sen_inter_df = pd.merge(sen_df, geometry_df, "left", "category")

    # def transform_to_localization_imagefolder(cls):
        # pass

    # def transform_to_change_detection_imagefolder(cls):
        # pass

    # @classmethod
    # def get_multiclass_classification_df_from_metadata(cls) -> pd.DataFrame:
        # df = cls.get_metadata_df()
        # df = df[df["label_str"] != "false_detection"]
        # for col in ("tl_0", "tl_1", "br_0", "br_1"):
            # df[f"super_bbox_{col}"] = df[f"bbox_{col}"]
        # df[["outer_bbox_tl_0", "outer_bbox_tl_1", "outer_bbox_br_0", "outer_bbox_br_1"]] = df.apply(cls.calculate_outer_bbox, axis = 1, result_type = "expand")
        # df[["inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]] = df.apply(cls.calculate_inner_bbox, axis = 1, result_type = "expand")
        # df = df.reset_index(drop = True)
        # return df

    # @classmethod
    # def get_multilabel_classification_df_from_metadata(cls) -> pd.DataFrame:
        # df = cls.get_metadata_df()
        # df = df[df["label_str"] != "false_detection"]
        # super_bboxes_df = df.groupby("image_path").agg({"bbox_tl_0": "min", "bbox_tl_1": "min", "bbox_br_0": "max", "bbox_br_1": "max"}).add_prefix("super_").reset_index(drop = False)
        # df = pd.merge(df, super_bboxes_df, how = "left", on = "image_path")
        # df[["outer_bbox_tl_0", "outer_bbox_tl_1", "outer_bbox_br_0", "outer_bbox_br_1"]] = df.apply(cls.calculate_outer_bbox, axis = 1, result_type = "expand")
        # df[["inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]] = df.apply(cls.calculate_inner_bbox, axis = 1, result_type = "expand")
        # df = df.reset_index(drop = True)
        # return df

    # @classmethod
    # def get_superresolution_df(cls):
        # pass

    # @classmethod
    # def get_change_detection_df_from_metadata(cls):
        # pass

# @classmethod
    # def plot_mcc_sample(cls, row: pd.Series):
        # PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        # image = iio.imread(cls.local/row["image_path"])
        # crop_tl, crop_br = row["crop_tl"], row["crop_br"]

        # dim, pre, post = row["pad_info"]
        # if pre or post:
            # if dim == 0:
                # image = np.pad(image, ((pre, post), (0, 0), (0, 0)))    
            # else:
                # image = np.pad(image, ((0, 0), (pre, post), (0, 0)))

        # cropped_image = image[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1], :]
        # crop_rect = get_rectangle_from_corners(crop_tl, crop_br, linewidth=2, edgecolor='b', facecolor='none')
        # #label_rect = get_rectangle_from_corners(row["label_tl"], row["label_br"], linewidth=2, edgecolor='r', facecolor='none')

        # fig = plt.figure(figsize = (6, 3), layout = "tight")
        # l = plt.subplot(121)
        # r = plt.subplot(122)
        # l.imshow(image)
        # l.add_patch(crop_rect)
        # r.imshow(cropped_image)
        # #r.add_patch(label_rect)
        # #ax.axis("off")
        # fig.savefig(f'fmow/{row["image_path"].stem}.png')
        # plt.cla()
        # fig.clf()
        # plt.close('all')
    
    # @classmethod
    # def transform_to_mcc_imagefolder(cls, row: pd.Series):
        # #imagefolder_path = get_new_dir(cls.local/"imagefolder")
        # #df = cls.get_mcc_dataset_df_from_imagefolder()

        # PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        # image = iio.imread(cls.local/row["image_path"])
        # crop_tl, crop_br = row["crop_tl"], row["crop_br"]

        # dim, pre, post = row["pad_info"]
        # if pre or post:
            # if dim == 0:
                # image = np.pad(image, ((pre, post), (0, 0), (0, 0)))    
            # else:
                # image = np.pad(image, ((0, 0), (pre, post), (0, 0)))
        # image = image[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1], :]
        # image = resize(image, (2000, 2000, 3), preserve_range=True, anti_aliasing=False)
        # image = image.astype(np.uint8)
        # iio.imwrite(get_new_dir(cls.temp_local/"imagefolder"/row["image_path"].parent)/row["image_path"].name, image, extension = ".jpg")
        # del image
    
    # @classmethod
    # def transform_to_mcc_hdf5(cls):
        # pass

    # @staticmethod
    # def get_corners_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            # #           bbox                     corners
            # #       (x, y, w, h)                (tl, br)
            # #  (x,y) -----w-------|      (y,x) -------------|
            # #    |                |        |                |
            # #    |                |        |                |
            # #    h                |   ->   |                |
            # #    |                |        |                |
            # #    |----------------|        | -------------(y,x)
            # "returns (y_min, x_min, y_max, x_max). y comes first because row-major."
            # return bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]
