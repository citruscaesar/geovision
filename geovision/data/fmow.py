from typing import Literal, Optional

import shutil
import h5py
import PIL
import json
import torch
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
from skimage.transform import resize

from .interfaces import Dataset, DatasetConfig
from geovision.io.local import FileSystemIO as fs

class FMoWETL:
    local_ms = Path.home()/"datasets"/"fmow-full"
    local_rgb = Path.home()/"datasets"/"fmow-rgb"
    local_sentinel = Path.home()/"datasets"/"fmow-sentinel"
    local_staging = Path("/run/media/sambhav/StorageHDD_1/datasets/fmow/")

    # fmt: off
    class_names = (
        'airport', 'airport_hangar', 'airport_terminal', 'amusement_park',
        'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint',
        'burial_site', 'car_dealership', 'construction_site', 'crop_field',
        'dam', 'debris_or_rubble', 'educational_institution',
        'electric_substation', 'factory_or_powerplant', 'fire_station',
        'flooded_road', 'fountain', 'gas_station', 'golf_course',
        'ground_transportation_station', 'helipad', 'hospital',
        'impoverished_settlement', 'interchange', 'lake_or_pond',
        'lighthouse', 'military_facility', 'multi-unit_residential',
        'nuclear_powerplant', 'office_building', 'oil_or_gas_facility',
        'park', 'parking_lot_or_garage', 'place_of_worship',
        'police_station', 'port', 'prison', 'race_track', 'railway_bridge',
        'recreational_facility', 'road_bridge', 'runway', 'shipyard',
        'shopping_mall', 'single-unit_residential', 'smokestack',
        'solar_farm', 'space_facility', 'stadium', 'storage_tank',
        'surface_mine', 'swimming_pool', 'toll_booth', 'tower',
        'tunnel_opening', 'waste_disposal', 'water_treatment_facility',
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
    def download_from_source(cls, dataset: Literal["rgb", "ms", "sentinel"] = "rgb"):
        """
        downloads fmow-{:dataset} from s3://spacenet-datasets/HostedDatasets/fmow/fmow-{:dataset} to :local_staging/:dataset
        cp train val test and seq directories, ignoring all .json files
        cp groundtruth.tar.bz2 and extract to :local_staging/:dataset/groundtruth/, removing _gt from test and seq dirs, mapping jsons
        """
        pass

    @classmethod
    def download_rgb_from_storage(cls, bucket = "s3://fmow/archives/rgb"):
        """
        downloads and extracts train.zip, val.zip, test.zip, seq.zip, groundtruth.tar.bz2 to :local_staging/rgb/
        """
        pass
    
    @classmethod
    def download_sentinel_from_storage(cls, bucket = "s3://fmow/archives/sentinel"):
        """
        downloads and extracts train.zip, val.zip, test.zip, groundtruth.zip to :local_staging/sentinel/
        """
        pass
    
    @classmethod
    def get_metadata_df(cls, dataset: Literal["rgb", "ms", "sentinel"] = "rgb"):
        """
        returns dataframe encapsulating groundtruth jsons into a single (georegistered) dataframe. looks for saved df in :local_staging/:dataset/metadata.h5. 
        if not found, it is generated using :staging/:dataset/groundtruth. raises OSError if :staging/groundtruth is not found or is empty
        """
        if dataset not in ("rgb", "ms", "sentinel"):
            raise ValueError(f"invalid :dataset, expected one of rgb, ms or sentinel, got {dataset}")
        try:
            metadata_df = pd.read_hdf(cls.local_staging / dataset / "metadata.h5", key = "dataset_df", mode = 'r')
        except OSError:
            #metadata_dir = get_valid_dir_err(cls.local_staging / dataset / "groundtruth", empty_ok=False)
            metadata_dir = fs.get_valid_dir_err(cls.local_rgb / "groundtruth", empty_ok=False)
            metadata = ["parent_dir", "label_str", "bbox"]
            metadata += [
                "img_filename", "gsd", "img_width", "img_height", "mean_pixel_height", "mean_pixel_width", "utm", "country_code", "cloud_cover", "timestamp",
                "scan_direction", "approximate_wavelengths", "catalog_id", "sensor_platform_name", "raw_location"#, "abs_cal_factors"
            ]
            metadata += [f"{x}{y}_dbl" for x in ("pan_resolution", "multi_resolution", "target_azimuth", "off_nadir_angle") for y in ("", "_start", "_end", "_min", "_max")]
            metadata += [f"{x}{y}_dbl" for x in ("sun_azimuth", "sun_elevation") for y in ("", "_min", "_max")]
            metadata = tuple(metadata)
            metadata_dict = {k:list() for k in metadata}
            for json_path in tqdm(metadata_dir.rglob("*_rgb.json"), total = 523846):
                with open(json_path) as json_file:
                    json_parsed = json.load(json_file)
                for bbox in json_parsed["bounding_boxes"][1:]:
                    metadata_dict["parent_dir"].append('/'.join(str(json_path).split('/')[-4:-1]))
                    metadata_dict["label_str"].append(bbox["category"])
                    metadata_dict["bbox"].append(bbox["box"])
                    for key in metadata[3:]:
                        metadata_dict[key].append(json_parsed[key])

            test_df = pd.read_json(cls.local_rgb /"groundtruth"/"test_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
            seq_df = pd.read_json(cls.local_rgb /"groundtruth"/"seq_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
            rename_dict = dict(zip(test_df["input"], test_df["output"])) | dict(zip(seq_df["input"], seq_df["output"]))

            def get_filename(parent_dir: str, img_filename: str):
                if parent_dir in rename_dict.keys():
                    parent_dir = rename_dict[parent_dir]
                    img_filename = '_'.join([parent_dir.split('/')[-1]] + img_filename.split('_')[-2:])
                return Path(parent_dir, img_filename).as_posix()

            metadata_df = (
                pd.DataFrame(metadata_dict)
                .assign(image_path = lambda df: df.apply(lambda x: get_filename(x["parent_dir"], x["img_filename"]), axis = 1))
                .assign(split = lambda df: df.apply(lambda x: str(x["image_path"]).split('/')[0], axis = 1))
                .sort_values(["split", "parent_dir", "timestamp"])
                .assign(bbox = lambda df: df["bbox"].apply(lambda x: cls.get_corners_from_bbox(x)))
                .assign(bbox_tl_0 = lambda df: df["bbox"].apply(lambda x: x[0]))
                .assign(bbox_tl_1 = lambda df: df["bbox"].apply(lambda x: x[1]))
                .assign(bbox_br_0 = lambda df: df["bbox"].apply(lambda x: x[2]))
                .assign(bbox_br_1 = lambda df: df["bbox"].apply(lambda x: x[3]))
                .drop(columns = ["split", "parent_dir", "img_filename", "bbox"])
                .reset_index(drop = True)
            )
            metadata_df = metadata_df[["image_path", "label_str", "img_height", "img_width", "bbox_tl_0", "bbox_tl_1", "bbox_br_0", "bbox_br_1"]+list(metadata[7:])]
            metadata_df.to_hdf(cls.local_staging / dataset / "metadata.h5", key = "dataset_df", mode = 'w', format = "fixed")

        metadata_df = (
            metadata_df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: Path(x)))
            .assign(approximate_wavelengths = lambda df: df["approximate_wavelengths"].astype(object))
            .assign(timestamp = lambda df: pd.to_datetime(df["timestamp"], format='mixed'))
        )
        return gpd.GeoDataFrame(metadata_df, geometry = gpd.GeoSeries.from_wkt(metadata_df["raw_location"]), crs = cls.crs).drop(columns = "raw_location")
    
    @classmethod
    def get_sentinel_metadata_df(cls):
        metadata_dir = cls.local_staging / "sentinel"
        try:
            metadata_df = pd.read_hdf(metadata_dir / "metadata.h5", key = "metadata", mode = 'r')
        except OSError:
            def get_image_path(row: pd.Series, split: str):
                return f"{split}/{row["category"]}/{row["category"]}_{row["location_id"]}/{row["category"]}_{row["location_id"]}_{row["image_id"]}.tif"
            train_df = (
                pd.read_csv(metadata_dir / "groundtruth" / "train.csv", index_col = 0)
                .assign(image_path = lambda df: df.apply(lambda x: get_image_path(x, "train"), axis = 1))
            )
            val_df = (
                pd.read_csv(metadata_dir / "groundtruth" / "val.csv", index_col = 0)
                .assign(image_path = lambda df: df.apply(lambda x: get_image_path(x, "val"), axis = 1))
            )
            test_df = (
                pd.read_csv(metadata_dir / "groundtruth" / "test.csv", index_col = 0)
                .assign(image_path = lambda df: df.apply(lambda x: get_image_path(x, "test"), axis = 1))
            )
            metadata_df = pd.concat([train_df, val_df, test_df], axis = 0).drop(columns = ["category", "location_id", "image_id"])
            metadata_df.to_hdf(metadata_dir / "metadata.h5", key = "metadata", mode = 'w')

        metadata_df = gpd.GeoDataFrame(metadata_df, geometry=gpd.GeoSeries.from_wkt(metadata_df["polygon"]), crs = 4326)
        metadata_df = metadata_df.drop(columns="polygon")
        metadata_df = metadata_df[["image_path", "timestamp", "geometry"]]
        return metadata_df 

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
        hdf5 = fs.get_new_dir(local, "hdf5") / f"fmow_{dataset}_multiclass.h5"

        with h5py.File(hdf5, mode = "w") as h5file:
            images = h5file.create_dataset("images", (len(df),), dtype = h5py.special_dtype(vlen=np.dtype("uint8")))
            for idx, row in tqdm(df.iterrows(), total = len(df)):
                image = iio.imread(staging / row["image_path"]).astype(np.uint8)
                image = image[row["outer_bbox_tl_0"]:row["outer_bbox_br_0"], row["outer_bbox_tl_1"]:row["outer_bbox_br_1"]]
                image = iio.imwrite('<bytes>', image, extension=".jpg")
                images[idx] = np.frombuffer(image, dtype = np.uint8)

        df["img_height"] = df.apply(lambda x: x["outer_bbox_br_0"] - x["outer_bbox_tl_0"], axis = 1)
        df["img_width"] = df.apply(lambda x: x["outer_bbox_br_1"] - x["outer_bbox_tl_1"], axis = 1)
        df = df[["image_path", "label_str", "img_width", "img_height", "inner_bbox_tl_0", "inner_bbox_tl_1", "inner_bbox_br_0", "inner_bbox_br_1"]]
        df.to_hdf(hdf5, key = "dataset_df", mode = "r+")
    
    def transform_to_superresolution_imagefolder(cls):
        def get_category(image_path: Path):
                image_path = str(image_path)
                category = '/'.join(image_path.split('/')[:-1])
                if category in rename_dict.keys():
                    return rename_dict[category]
                return category

        test_df = pd.read_json(FMoW.local_rgb/"groundtruth"/"test_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("test_gt", "test")))
        seq_df = pd.read_json(FMoW.local_rgb/"groundtruth"/"seq_gt_mapping.json").assign(input = lambda df: df["input"].apply(lambda x: str(x).replace("seq_gt", "seq")))
        rename_dict = dict(zip(test_df["output"], test_df["input"])) | dict(zip(seq_df["output"], seq_df["input"]))

        sen_df = FMoW.get_sentinel_metadata_df()
        sen_df["timestamp"] = pd.to_datetime(sen_df["timestamp"], format = "mixed")
        sen_df["category"] = sen_df["image_path"].apply(lambda x: '/'.join(x.split('/')[:-1]))

        rgb_df = (
            FMoW.get_metadata_df()
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
        image_dims = row["img_height"], row["img_width"]
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
    
class FMoWHDF5Classification(Dataset):
    name = "fmow_hdf5_classification"
    class_names = FMoWETL.class_names
    num_classes = FMoWETL.num_classes
    means = FMoWETL.means
    std_devs = FMoWETL.std_devs

    df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce = True),
        "split": pa.Column(str, pa.Check.isin(Dataset.splits)),
    }, index = pa.Index(int, unique = True))

    split_df_schema = pa.DataFrameSchema({
        "image_path": pa.Column(str, coerce=True),
        "df_idx": pa.Column(int),
        "label_idx": pa.Column(int, pa.Check.isin(tuple(range(0, FMoWETL.num_classes))))
    }, index = pa.Index(int, unique = True))

    def __init__(self, split: Literal["train", "val", "trainval", "test", "all"] = "all", config: Optional[DatasetConfig] = None):
        self._root = fs.get_valid_file_err(FMoWETL.local_rgb, "hdf5", "fmow_rgb_multiclass.h5")
        self._split = self.get_valid_split_err(split)
        self._config = config or FMoWETL.default_config
        self._df = self._config.verify_and_get_df(schema = self.df_schema, fallback_df = pd.read_hdf(self._root, key = "dataset_df", mode = 'r'))
        self._split_df = self._config.verify_and_get_split_df(df = self._df, schema = self.df_schema, split = self._split)
        PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
    
    def __len__(self) -> int:
        return len(self._split_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        with h5py.File(self.root, mode="r") as f:
            image = iio.imread(BytesIO(f["images"][idx_row["df_idx"]]))
        image = self._config.image_pre(image)
        if self._split in ("train", "trainval", "all"):
            image = self._config.train_aug(image)
        elif self._split in ("val", "test"):
            image = self._config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

    @property
    def root(self):
        return self._root

    @property
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        return self._split

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        return self._split_df