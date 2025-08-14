from typing import Literal, Optional

import os 
import h5py
import PIL
import json
import torch
import shutil
import zipfile
import tarfile
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import pandera as pa
import imageio.v3 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from matplotlib.patches import Rectangle 
from torchvision.transforms import v2 as T

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
            test_frac = 0.2,
            val_frac = 0.1
        ),
        image_pre = T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale = True)
        ]), 
        target_pre = None, 
        train_aug = T.Compose([
            T.Resize(512),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(0.5),
        ]), 
        eval_aug = T.TenCrop(256, vertical_flip = True), 
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
        src: Literal["archive", "imagefolder", "hdf5"],
        subset: Literal["rgb", "ms", "sen2"]
    ) -> pd.DataFrame:
        
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

        if subset in ("ms", "rgb"):
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

            if src == "archive":
                archive_path = fs.get_valid_file_err(cls.local, "staging", "rgb", "groundtruth.tar.bz2")
                try:
                    raise OSError
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
                    for json_path in tqdm(list(groundtruth_path.rglob(f"*_{"msrgb" if subset == "ms" else "rgb"}.json"))):
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
        src: Literal["archive", "imagefolder"], 
        to: Literal["imagefolder", "hdf5"], 
        subset: Literal["rgb_clf", "sen2_super_res"],
        crop_to_bbox: bool = False,
        resize_to: Optional[int] = None,
        jpeg_quality: int = 90
    ):
        assert src in ("archive", "imagefolder")
        assert to in ("imagefolder", "hdf5")
        assert subset in ("rgb_clf", "sen2_super_res")

        if subset == "rgb_clf":
            # Assert the source files exist, as archive or imagefolder
            # Load the dataframes from the source directories metadata
            # Check :crop_to_bbox is true, load the spatial dataframe, take the union of all bboxes per image, apply square crop and update spatial geometry.
            # Check if :resize_to is not None, resize and pad all images to :(int)resize_to. This renders geometry meaningless, and is purely for fast pre-training, so remove the irrelevant columns.
            # To convert archive to imagefolder, extract the archives and update the images inplace, rather than extract-process-save one by one. This will require more space initially
            # To convert archive/imagefolder to hdf5, load each image, crop and pad as required, encode to jpeg and save to {subset}.h5
            pass

        elif subset == "sen2_super_res":
            # Assert the source files exist, as archive or imagefolder
            # Load the dataframes from the source directories metadata
            # Create the backwards rename dict using test_index and seq_index dataframes from the metadata
            # Assert crop_to_bbox is False and resize_to is None.
            # Create a injective mapping between rgb and sen2_ms, by filtering samples by time and category
            # ... understand the geometry mapping b/w the two
            pass

    # Transformations
    @classmethod
    def transform_to_classification_imagefolder(cls, dataset: Literal["rgb", "ms"] = "rgb", classification: Literal["multiclass", "multilabel"] = "multiclass"):
        if dataset == "rgb":
            local, staging = cls.local_rgb, cls.local_staging / "rgb"
        elif dataset == "ms":
            local, staging = cls.local_ms, cls.local_staging / "ms"
        else:
            raise ValueError(f"invalid :dataset, expected either rgb or ms, got {dataset}")

        if classification == "multiclass":
            df = cls.get_multiclass_classification_df_from_metadata()
        elif classification == "multilabel":
            df = cls.get_multilabel_classification_df_from_metadata()
        else:
            raise ValueError(f"invalid :classification, expected either multiclass or multilabel, got {dataset}")

        imagefolder = fs.get_new_dir(local, "imagefolder")

        for subdir in df["image_path"].apply(lambda x: x.parent).unique():
            fs.get_new_dir(imagefolder, subdir)

        for _, row in tqdm(df.iterrows(), total = len(df)):
            if row["img_height"] >= 2000 or row["img_width"] >= 2000:
                iio.imwrite(
                    uri = imagefolder/row["image_path"],
                    image = iio.imread(staging / row["image_path"])[row["outer_bbox_tl_0"]:row["outer_bbox_br_0"], row["outer_bbox_tl_1"]:row["outer_bbox_br_1"]].astype(np.uint8),
                    extension = ".jpg"
                )
            else:
                shutil.copy(imagefolder/row["image_path"], staging/row["image_path"])

        df["img_height"] = df.apply(lambda x: x["outer_bbox_br_0"] - x["outer_bbox_tl_0"], axis = 1)
        df["img_width"] = df.apply(lambda x: x["outer_bbox_br_1"] - x["outer_bbox_tl_1"], axis = 1)
        df = df[["image_path", "label_str", "img_width", "img_height", "inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]]
        df.to_hdf(imagefolder/"dataset.h5", key = "dataset_df", mode = "w")

    @classmethod
    def transform_to_multiclass_classification_hdf5(cls, dataset: Literal["rgb", "ms"] = "rgb"):
        PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        if dataset == "rgb":
            local, staging = cls.local_rgb, cls.local_staging / "rgb"
        elif dataset == "ms":
            local, staging = cls.local_ms, cls.local_staging / "ms"
        else:
            raise ValueError(f"invalid :dataset, expected either rgb or ms, got {dataset}")

        df = cls.get_multiclass_classification_df_from_metadata()
        ds = fs.get_new_dir(local, "hdf5") / f"fmow_{dataset}_multiclass.h5"

        with h5py.File(ds, mode = "w") as f:
            images = f.create_dataset("images", shape = len(df), dtype = h5py.special_dtype(vlen=np.uint8))
            for idx, row in tqdm(df.iterrows(), total = len(df)):
                image = iio.imread(staging / row["image_path"]).astype(np.uint8)
                image = image[row["outer_bbox_tl_0"]:row["outer_bbox_br_0"], row["outer_bbox_tl_1"]:row["outer_bbox_br_1"]]
                image = iio.imwrite('<bytes>', image, extension=".jpg")
                images[idx] = np.frombuffer(image, dtype = np.uint8)

        df["img_height"] = df.apply(lambda x: x["outer_bbox_br_0"] - x["outer_bbox_tl_0"], axis = 1)
        df["img_width"] = df.apply(lambda x: x["outer_bbox_br_1"] - x["outer_bbox_tl_1"], axis = 1)
        df = df[["image_path", "label_str", "img_width", "img_height", "inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]]
        df.to_hdf(ds, key = "index", mode = "r+")
    
    def transform_to_superresolution_imagefolder(cls):
        def get_category(image_path: Path):
            image_path = str(image_path)
            category = '/'.join(image_path.split('/')[:-1])
            if category in rename_dict.keys():
                return rename_dict[category]
            return category

        test_df = pd.read_json(cls.local_rgb/"groundtruth"/"test_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
        seq_df = pd.read_json(cls.local_rgb/"groundtruth"/"seq_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
        rename_dict = dict(zip(test_df["output"], test_df["input"])) | dict(zip(seq_df["output"], seq_df["input"]))

        sen_df = cls.get_sentinel_metadata_df()
        sen_df["timestamp"] = pd.to_datetime(sen_df["timestamp"], format = "mixed")
        sen_df["category"] = sen_df["image_path"].apply(lambda x: '/'.join(x.split('/')[:-1]))

        rgb_df = (
            cls.get_metadata_df()
            .assign(split = lambda df: df["image_path"].apply(lambda x: str(x).split('/')[0]))
            .assign(category = lambda df: df["image_path"].apply(lambda x: get_category(x)))
        )
        rgb_df = rgb_df[rgb_df["label_str"] != "false_detection"]
        rgb_df = rgb_df[rgb_df["split"] != "seq"]

        time_range_df = rgb_df.groupby("category").agg({"timestamp": ["min", "max"]}).reset_index(drop = False)
        time_range_df.columns = ["category", "timestamp_min", "timestamp_max"]
        sen_df = pd.merge(sen_df, time_range_df, how="left", on="category")
        sen_df = sen_df[sen_df.apply(lambda x: x["timestamp"] > x["timestamp_min"] and x["timestamp"] < x["timestamp_max"], axis = 1)]

        sen_categories = set(sen_df["category"].unique())

        rgb_df = rgb_df[rgb_df["category"].apply(lambda x: x in sen_categories)]
        rgb_df = rgb_df[["image_path", "timestamp", "geometry", "category"]].sort_values(by = ["category", "timestamp"], ascending=[True, True]).reset_index(drop = True)
        sen_df = sen_df.drop(columns = ["timestamp_min", "timestamp_max"]).sort_values(by = ["category", "timestamp"], ascending=[True, True]).reset_index(drop = True)

        geometry_df = pd.concat([rgb_df, sen_df], axis = 0).groupby("category").agg({"geometry": shapely.intersection_all}).reset_index(drop = False)
        geometry_df.columns = ["category", "intersection"]

        rgb_inter_df = pd.merge(rgb_df, geometry_df, "left", "category")
        sen_inter_df = pd.merge(sen_df, geometry_df, "left", "category")

    def transform_to_localization_imagefolder(cls):
        pass

    def transform_to_change_detection_imagefolder(cls):
        pass

    @classmethod
    def get_multiclass_classification_df_from_metadata(cls) -> pd.DataFrame:
        df = cls.get_metadata_df()
        df = df[df["label_str"] != "false_detection"]
        for col in ("tl_0", "tl_1", "br_0", "br_1"):
            df[f"super_bbox_{col}"] = df[f"bbox_{col}"]
        df[["outer_bbox_tl_0", "outer_bbox_tl_1", "outer_bbox_br_0", "outer_bbox_br_1"]] = df.apply(cls.calculate_outer_bbox, axis = 1, result_type = "expand")
        df[["inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]] = df.apply(cls.calculate_inner_bbox, axis = 1, result_type = "expand")
        df = df.reset_index(drop = True)
        return df

    @classmethod
    def get_multilabel_classification_df_from_metadata(cls) -> pd.DataFrame:
        df = cls.get_metadata_df()
        df = df[df["label_str"] != "false_detection"]
        super_bboxes_df = df.groupby("image_path").agg({"bbox_tl_0": "min", "bbox_tl_1": "min", "bbox_br_0": "max", "bbox_br_1": "max"}).add_prefix("super_").reset_index(drop = False)
        df = pd.merge(df, super_bboxes_df, how = "left", on = "image_path")
        df[["outer_bbox_tl_0", "outer_bbox_tl_1", "outer_bbox_br_0", "outer_bbox_br_1"]] = df.apply(cls.calculate_outer_bbox, axis = 1, result_type = "expand")
        df[["inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]] = df.apply(cls.calculate_inner_bbox, axis = 1, result_type = "expand")
        df = df.reset_index(drop = True)
        return df

    @classmethod
    def get_superresolution_df(cls):
        pass

    @classmethod
    def get_change_detection_df_from_metadata(cls):
        pass

    @staticmethod
    def get_corners_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            #           bbox                     corners
            #       (x, y, w, h)                (tl, br)
            #  (x,y) -----w-------|      (y,x) -------------|
            #    |                |        |                |
            #    |                |        |                |
            #    h                |   ->   |                |
            #    |                |        |                |
            #    |----------------|        | -------------(y,x)
            "returns (y_min, x_min, y_max, x_max). y comes first because row-major."
            return bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]

    @staticmethod
    def get_rectangle_from_corners(tl: tuple[int, int], br: tuple[int, int], **kwargs) -> Rectangle:
            #          corners                   rect 
            #         (tl, br)                ((x,y), w, h)
            #   (y,x) ------------|        |----------------|
            #    |                |        |                |
            #    |                |        h                |
            #    |                |   ->   |                |
            #    |                |        |                |
            #    |--------------(y,x)      (x,y) -----w-----|
            return Rectangle((tl[1], br[0]), br[1] - tl[1], tl[0] - br[0], **kwargs)
    
    @staticmethod
    def calculate_outer_bbox(row: pd.Series) -> tuple[int, int, int, int]:
        # by row-major convention - dim_0: height, dim_1: width
        image_dims = row["image_height"], row["image_width"]
        bbox_tl, bbox_br = (row["super_bbox_tl_0"], row["super_bbox_tl_1"]), (row["super_bbox_br_0"], row["super_bbox_br_1"])
        bbox_dims = bbox_br[0] - bbox_tl[0], bbox_br[1] - bbox_tl[1]

        # bbox will be modified along the smaller dimension of the bbox, sdim
        sdim = np.argmin(bbox_dims)
        ldim = np.abs(sdim - 1)

        # skip over images smaller than 2000x2000
        if image_dims[ldim] < 2000:
            return 0, 0, *image_dims
        
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

    @staticmethod
    def calculate_inner_bbox(row: pd.Series) -> tuple[int, int, int, int]:
        return (
            row["bbox_tl_0"] - row["outer_bbox_tl_0"],
            row["bbox_tl_1"] - row["outer_bbox_tl_1"],
            row["bbox_br_0"] - row["outer_bbox_tl_0"],
            row["bbox_br_1"] - row["outer_bbox_tl_1"],
        )

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
    
# class FMoWHDF5Classification(Dataset):
    # name = "fmow_hdf5_classification"
    # class_names = FMoWETL.class_names
    # num_classes = FMoWETL.num_classes
    # means = FMoWETL.means
    # std_devs = FMoWETL.std_devs

    # df_schema = pa.DataFrameSchema({
        # "image_path": pa.Column(str, coerce = True),
        # "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
    # }, index = pa.Index(int, unique = True))

    # split_df_schema = pa.DataFrameSchema({
        # "image_path": pa.Column(str, coerce=True),
        # "df_idx": pa.Column(int),
        # "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, FMoWETL.num_classes))))
    # }, index = pa.Index(int, unique = True))

    # def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        # self._root = fs.get_valid_file_err(FMoWETL.local_rgb, "hdf5", "fmow_rgb_multiclass.h5")
        # self._split = self.get_valid_split_err(split)
        # self._config = config or FMoWETL.default_config
        # self._df = self._config.verify_and_get_df(schema = self.df_schema, fallback_df = pd.read_hdf(self._root, key = "dataset_df", mode = 'r'))
        # self._split_df = self._config.verify_and_get_split_df(df = self._df, schema = self.df_schema, split = self._split)
        # PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
    
    # def __len__(self) -> int:
        # return len(self._split_df)
    
    # def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        # idx_row = self.split_df.iloc[idx]
        # with h5py.File(self.root, mode="r") as f:
            # image = iio.imread(BytesIO(f["images"][idx_row["df_idx"]]))
        # image = self._config.image_pre(image)
        # if self._split in ("train", "trainvaltest"):
            # image = self._config.train_aug(image)
        # elif self._split in ("val", "test"):
            # image = self._config.eval_aug(image)
        # return image, idx_row["label_idx"], idx_row["df_idx"]

    # @property
    # def root(self):
        # return self._root

    # @property
    # def split(self) -> Literal["train", "val", "test", "trainvaltest", "all"]:
        # return self._split

    # @property
    # def df(self) -> pd.DataFrame:
        # return self._df

    # @property
    # def split_df(self) -> pd.DataFrame:
        # return self._split_df