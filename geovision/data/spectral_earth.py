from typing import Literal, Optional, Callable, Sequence
from numpy.typing import NDArray

import h5py
import py7zr
import torch
import shutil
import zipfile
import subprocess

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
import imageio.v3 as iio  # noqa: F401
import pandera.pandas as pa
import torchvision.transforms.v2 as T

from tqdm import tqdm
from pathlib import Path
from affine import Affine
from shapely import Polygon
from rasterio.crs import CRS
from rasterio.features import sieve, shapes
from multiprocessing import Pool, cpu_count
from matplotlib.colors import ListedColormap
from geovision.io.local import FileSystemIO as fs 
from torchvision.datasets.utils import download_url
from skimage.measure import find_contours, approximate_polygon

from geovision.data import Dataset, DatasetConfig
from geovision.data.transforms import SegmentationCompose

import logging
logger = logging.getLogger(__name__)

# image_patches: L2A reflectance (orthorectified, atmospherically corrected, absorption bands removed, 202x128x128, little-endian int16) 

class SpectralEarth:
    local: Path = Path.home() / "datasets" / "spectral_earth"
    num_bands = 202 # after removing water absorption bands

    # Cropland Data Layer
    cdl_classes = (
        'Corn', 'Cotton', 'Rice', 'Sorghum', 'Soybeans', 'Sunflower', 'Sugarcane', 'Tomatoes', 'Grapes', 'Citrus', 'Almonds', 'Walnuts',
        'Pistachios', 'Prunes'
    )
    cdl_remap_dict = {
        1: 0, # Corn
        12: 0, # Sweet Corn -> Corn
        13: 0, # Pop or Orn Corn -> Corn
        2: 1, # Cotton
        3: 2, # Rice
        4: 3, # Sorghum
        236: 3, # DblCrop WinWht/Sorghum -> Sorghum
        5: 4, # Soybeans
        6: 5, # Sunflower
        45: 6, # Sugarcane
        54: 7, # Tomatoes
        69: 8, # Grapes
        72: 9, # Citrus
        212: 9, # Oranges -> Citrus
        75: 10, # Almonds
        76: 11, # Walnuts
        204: 12, # Pistachios
        210: 13, # Prunes
        220: 13, # Plums -> Prunes
        # Everything else -> Background (255)
    }

    cdl_colors = {
        0: "#ffd400",  # Corn - Bright golden yellow
        1: "#ff2626",  # Cotton - Bright red
        2: "#00a9e6",  # Rice - Cyan blue
        3: "#ff9e0f",  # Sorghum - Orange
        4: "#267300",  # Soybeans - Dark green
        5: "#ffff00",  # Sunflower - Yellow
        6: "#b27fff",  # Sugarcane - Purple
        7: "#f2a377",  # Tomatoes - Peachy-orange
        8: "#550070",  # Grapes - Deep purple
        9: "#ff6666",  # Citrus - Red-pink
        10: "#702600", # Almonds - Dark brown
        11: "#00a582", # Walnuts - Teal
        12: "#702600", # Pistachios - Dark brown
        13: "#702600", # Prunes - Dark brown
        255: "#000000", # Background - Black
    }

    # National Land Cover Database (NLCD)
    nlcd_classes = (
        'Open Water', 'Developed, Open Space', 'Developed, Low Intensity', 'Developed, Medium Intensity', 'Developed, High Intensity',
        'Barren Land', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Shrub/Scrub', 'Grassland/Herbaceous', 'Pasture/Hay',
        'Cultivated Crops', 'Woody Wetlands', 'Emergent Herbaceous Wetlands'
    )
    nlcd_remap_dict = {
        11: 0,   # Open Water
        21: 1,   # Developed, Open Space
        22: 2,   # Developed, Low Intensity
        23: 3,   # Developed, Medium Intensity
        24: 4,   # Developed, High Intensity
        31: 5,   # Barren Land
        41: 6,   # Deciduous Forest
        42: 7,   # Evergreen Forest
        43: 8,   # Mixed Forest
        52: 9,   # Shrub/Scrub
        71: 10,  # Grassland/Herbaceous
        81: 11,  # Pasture/Hay
        82: 12,  # Cultivated Crops
        90: 13,  # Woody Wetlands
        95: 14,  # Emergent Herbaceous Wetlands

        12: 255, # Perennial Ice/Snow -> Background
    }

    nlcd_colors = {
        0: "#466b9f",   # Open Water - Deep blue
        1: "#dec5c5",   # Developed, Open Space - Light pink-gray
        2: "#d99282",   # Developed, Low Intensity - Salmon pink
        3: "#eb0000",   # Developed, Medium Intensity - Bright red
        4: "#ab0000",   # Developed, High Intensity - Dark red
        5: "#b3ac9f",   # Barren Land - Gray-tan
        6: "#68ab5f",   # Deciduous Forest - Medium green
        7: "#1c5f2c",   # Evergreen Forest - Dark green
        8: "#b5c58f",   # Mixed Forest - Light olive green
        9: "#ccb879",   # Shrub/Scrub - Tan-gold
        10: "#dfdfc2",  # Grassland/Herbaceous - Light yellow-tan
        11: "#dcd939",  # Pasture/Hay - Yellow-green
        12: "#ab6c28",  # Cultivated Crops - Brown-orange
        13: "#b8d9eb",  # Woody Wetlands - Light cyan
        14: "#6c9fb8",  # Emergent Herbaceous Wetlands - Medium blue-gray
        255: "#000000", # Background - Black
    }

    # CORINE Land Cover (BigEarthNet 19-class nomenclature)
    corine_classes = (
        'Urban fabric', 'Industrial or commercial units', 'Arable land', 'Permanent crops', 'Pastures', 'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Agro-forestry areas',
        'Broad-leaved forest', 'Coniferous forest', 'Mixed forest', 'Natural grassland and sparsely vegetated areas',
        'Moors, heathland and sclerophyllous vegetation', 'Transitional woodland, shrub', 'Beaches, dunes, sands', 'Inland wetlands',
        'Coastal wetlands', 'Inland waters', 'Marine waters'
    )
    corine_remap_dict = {
        # Artificial surfaces
        111: 0,  # Continuous urban fabric -> Urban fabric
        112: 0,  # Discontinuous urban fabric -> Urban fabric
        121: 1,  # Industrial or commercial units -> Industrial or commercial units

        # Agricultural areas
        211: 2,  # Non-irrigated arable land -> Arable land
        212: 2,  # Permanently irrigated land -> Arable land
        213: 2,  # Rice fields -> Arable land
        221: 3,  # Vineyards -> Permanent crops
        222: 3,  # Fruit trees and berry plantations -> Permanent crops
        223: 3,  # Olive groves -> Permanent crops
        241: 3,  # Annual crops associated with permanent crops -> Permanent crops
        231: 4,  # Pastures -> Pastures
        242: 5,  # Complex cultivation patterns -> Complex cultivation patterns
        243: 6,  # Land principally occupied by agriculture -> Agriculture with natural vegetation
        244: 7,  # Agro-forestry areas -> Agro-forestry areas

        # Forest and semi-natural areas
        311: 8,  # Broad-leaved forest -> Broad-leaved forest
        312: 9,  # Coniferous forest -> Coniferous forest
        313: 10, # Mixed forest -> Mixed forest
        321: 11, # Natural grasslands -> Natural grassland and sparsely vegetated areas
        333: 11, # Sparsely vegetated areas -> Natural grassland and sparsely vegetated areas
        322: 12, # Moors and heathland -> Moors, heathland and sclerophyllous vegetation
        323: 12, # Sclerophyllous vegetation -> Moors, heathland and sclerophyllous vegetation
        324: 13, # Transitional woodland-shrub -> Transitional woodland, shrub
        331: 14, # Beaches, dunes, sands -> Beaches, dunes, sands

        # Wetlands
        411: 15, # Inland marshes -> Inland wetlands
        412: 15, # Peat bogs -> Inland wetlands
        421: 16, # Salt marshes -> Coastal wetlands
        423: 16, # Intertidal flats -> Coastal wetlands

        # Water bodies
        511: 17, # Water courses -> Inland waters
        512: 17, # Water bodies -> Inland waters
        521: 18, # Coastal lagoons -> Marine waters
        522: 18, # Estuaries -> Marine waters
        523: 18, # Sea and ocean -> Marine waters

        # Removed classes (mapped to background)
        122: 255, # Road and rail networks and associated land -> Background
        123: 255, # Port areas -> Background
        124: 255, # Airports -> Background
        131: 255, # Mineral extraction sites -> Background
        132: 255, # Dump sites -> Background
        133: 255, # Construction sites -> Background
        141: 255, # Green urban areas -> Background
        142: 255, # Sport and leisure facilities -> Background
        332: 255, # Bare rocks -> Background
        334: 255, # Burnt areas -> Background
        335: 255, # Glaciers and perpetual snow -> Background
        422: 255, # Salines -> Background
    }

    corine_colors = {
        0: "#ff0000",   # Urban fabric - Red
        1: "#cc4df2",   # Industrial or commercial units - Purple
        2: "#ffffa8",   # Arable land - Pale yellow
        3: "#e68000",   # Permanent crops - Orange
        4: "#e6e64d",   # Pastures - Light yellow
        5: "#ffe6a6",   # Complex cultivation patterns - Cream
        6: "#e6cc4d",   # Agriculture with natural vegetation - Tan
        7: "#f2cca6",   # Agro-forestry areas - Beige
        8: "#80ff00",   # Broad-leaved forest - Bright green
        9: "#00a600",   # Coniferous forest - Dark green
        10: "#4dff00",  # Mixed forest - Medium green
        11: "#ccf24d",  # Natural grassland and sparsely vegetated - Yellow-green
        12: "#a6ffe6",  # Moors, heathland and sclerophyllous - Cyan-green
        13: "#a6f200",  # Transitional woodland, shrub - Lime
        14: "#e6e6e6",  # Beaches, dunes, sands - Light gray
        15: "#a6a6ff",  # Inland wetlands - Light blue
        16: "#ccccff",  # Coastal wetlands - Pale blue
        17: "#00ccf2",  # Inland waters - Cyan
        18: "#e6f2ff",  # Marine waters - Very pale blue
        255: "#000000", # Background - Black
    }

    index_df_schema = pa.DataFrameSchema(
        columns={
            "image_name": pa.Column(str, coerce=True),
        },
        index=pa.Index(int)
    )

    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="equal",
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=T.Compose([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.Identity()
    )   

    @classmethod
    def extract(cls): ...

    @classmethod
    def download(cls): ...

    @classmethod
    def load(
            cls, 
            table: Literal["index"],
            src: Literal["archive", "imagefolder", "hdf5"],
            subset: Literal["spectral_earth_nlcd", "spectral_earth_cdl", "spectral_earth_corine", "spectral_earth_unsup"]
        ) -> pd.DataFrame:

        assert src in ("staging", "imagefolder", "hdf5")
        assert subset in ("spectral_earth_nlcd", "spectral_earth_cdl", "spectral_earth_corine", "spectral_earth_unsup")

        if src == "hdf5":
            return pd.read_hdf(cls.local/"hdf5"/f"{subset}.h5", key = table, mode = 'r')
        
        elif src == "staging":
            return pd.read_hdf(fs.get_valid_dir_err(cls.local, "staging")/"metadata.h5", key = table, mode = 'r')

        elif src == "imagefolder":
            imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder")
            try:
                return pd.read_hdf(imagefolder_path/"metadata.h5", key = f"{subset.split('_')[-1]}/{table}", mode = 'r')#[:1000]

            except OSError:
                assert table in ("index"), \
                    f"not implemented error, expected :table to be index, since this function cannot create {table} metadata"
                    
                df = pd.DataFrame(imagefolder_path.rglob("*.tif"), columns = ["image_path"])
                df["dataset"] = df["image_path"].astype(str).str.split('/').str[-3]
                df["image_name"] = df["image_path"].astype(str).str.split('/').str[-2:].str.join('/').str.removesuffix('.tif')
                df = df.drop(columns = "image_path")

                cdl_df, corine_df, _, nlcd_df = (d.reset_index(drop=True).drop(columns="dataset") for _, d in df.groupby("dataset", sort = True))

                cdl_df.to_hdf(imagefolder_path/"metadata.h5", key = "cdl/index", mode = "w")
                corine_df.to_hdf(imagefolder_path/"metadata.h5", key = "corine/index", mode = "r+")
                nlcd_df.to_hdf(imagefolder_path/"metadata.h5", key = "nlcd/index", mode = "r+")

                if subset == "cdl":
                    return cdl_df
                elif subset == "corine":
                    return corine_df
                elif subset == "nlcd":
                    return nlcd_df
                
    @classmethod
    def transform(cls, to: Literal["imagefolder", "hdf5"], subset: Literal["spectral_earth_nlcd", "spectral_earth_cdl", "spectral_earth_corine", "spectral_earth_unsup"]): 
        assert to in ("imagefolder", "hdf5")
        assert subset in ("spectral_earth_nlcd", "spectral_earth_cdl", "spectral_earth_corine", "spectral_earth_unsup") 
    
        if to == "imagefolder":
            ...
            # move to :local/imagefolder/cdl/images,masks and so on for each subset

        elif to == "hdf5": 
            imagefolder_path = fs.get_valid_dir_err(cls.local, "imagefolder") 
            hdf5_path = fs.get_new_dir(cls.local / "hdf5")
            index_df = cls.load('index', 'imagefolder', subset)
            remap_fn = cls.get_remap_fn(subset.removeprefix("spectral_earth_"))
            cls._write_to_hdf(subset, index_df, remap_fn, imagefolder_path, hdf5_path)

            #args = [(hdf5_path, imagefolder_path, x, ) for x in ("spectral_earth_cdl", "spectral_earth_nlcd", "spectral_earth_corine")]
            #with Pool(processes=cpu_count()) as pool:
                #pool.starmap(cls._write_to_hdf, args)
    
    @staticmethod
    def _write_to_hdf(name: str, index_df: pd.DataFrame, remap_fn: Callable, imagefolder_path: Path, hdf5_path: Path):
        index_df.to_hdf(hdf5_path / f"{name}.h5", key = "index", mode = "w")

        with h5py.File(name = hdf5_path / f"{name}.h5", mode = 'r+') as f:
            images: h5py.Dataset = f.create_dataset("images", shape = (len(index_df), 202, 128, 128), dtype = np.int16)
            masks: h5py.Dataset = f.create_dataset("masks", shape = (len(index_df), 128, 128), dtype = np.uint8)
            cdl_masks: h5py.Dataset = f.create_dataset("cdl_masks", shape = (len(index_df), 128, 128), dtype = np.uint8)

            for idx, row in tqdm(index_df.iterrows(), total = len(index_df), desc = f"encoding {name}"):
                images[idx] = iio.imread(imagefolder_path/"enmap"/f"{row["image_name"]}.tif").transpose(2,0,1)
                mask = iio.imread(imagefolder_path/name.split('_')[-1]/f"{row["image_name"]}.tif").squeeze()
                cdl_masks[idx], masks[idx] = mask, remap_fn(mask)

    @classmethod
    def get_colormap(cls, dataset: Literal["cdl", "corine", "nlcd"]) -> ListedColormap:
        if dataset == "cdl":
            return ListedColormap([x for (_, x) in sorted(cls.cdl_colors.items())], name = "cdl", N = 15)
        elif dataset == "nlcd":
            return ListedColormap([x for (_, x) in sorted(cls.nlcd_colors.items())], name = "nlcd", N = 16)
        elif dataset == "corine":
            return ListedColormap([x for (_, x) in sorted(cls.corine_colors.items())], name = "corine", N = 20)

    @classmethod
    def get_remap_fn(cls, dataset: Literal["cdl", "corine", "nlcd"]) -> Callable:
        if dataset == "cdl":
            return np.vectorize(lambda x: cls.cdl_remap_dict.get(x, 255))
        elif dataset == "nlcd":
            return np.vectorize(lambda x: cls.nlcd_remap_dict.get(x, 255))
        elif dataset == "corine":
            return np.vectorize(lambda x: cls.corine_remap_dict.get(x, 255))

                
class SpectralEarth_CDL_Segmentation_HDF5(Dataset):
    name = "spectral_earth_cdl"
    task = "segmentation"
    subtask = "semantic"
    storage = "hdf5"
    class_names = SpectralEarth.cdl_classes
    num_classes = len(SpectralEarth.cdl_classes)
    root = SpectralEarth.local/"hdf5"/"spectral_earth_cdl.h5"
    schema = SpectralEarth.index_df_schema
    config = SpectralEarth.default_config
    loader = SpectralEarth.load
    metadata_group_prefix = None

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.index_df.assign(df_idx = lambda df: df.index)
        #self.df = self.get_df(prefix_root_to_paths=False)
   
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = f["masks"][idx_row["df_idx"]]
            # mask does not need one hot encoding. it's only done for BinaryCrossEntropy or Dice

        if self.split != "all":
            image = image.transpose(1,2,0)
            image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        return image, mask, idx_row["df_idx"]


class SpectralEarth_NLCD_Segmentation_HDF5(Dataset):
    name = "spectral_earth_nlcd"
    task = "segmentation"
    subtask = "semantic"
    storage = "hdf5"
    class_names = SpectralEarth.nlcd_classes
    num_classes = len(SpectralEarth.nlcd_classes)
    root = SpectralEarth.local/"hdf5"/"spectral_earth_nlcd.h5"
    schema = SpectralEarth.index_df_schema
    config = SpectralEarth.default_config
    loader = SpectralEarth.load
    metadata_group_prefix = None

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.index_df.assign(df_idx = lambda df: df.index)
        #self.df = self.get_df(prefix_root_to_paths=False)
   
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = f["masks"][idx_row["df_idx"]]
            # mask does not need one hot encoding. it's only done for BinaryCrossEntropy or Dice

        if self.split != "all":
            image = image.transpose(1,2,0)
            image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        return image, mask, idx_row["df_idx"]


class SpectralEarth_CORINE_Segmentation_HDF5(Dataset):
    name = "spectral_earth_corine"
    task = "segmentation"
    subtask = "semantic" # TODO: multilabel semantic segmentation?
    storage = "hdf5"
    class_names = SpectralEarth.corine_classes
    num_classes = len(SpectralEarth.corine_classes)
    root = SpectralEarth.local/"hdf5"/"spectral_earth_corine.h5"
    schema = SpectralEarth.index_df_schema
    config = SpectralEarth.default_config
    loader = SpectralEarth.load
    metadata_group_prefix = None

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.index_df.assign(df_idx = lambda df: df.index)
   
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx_row = self.df.iloc[idx]
        with h5py.File(self.root, mode = "r") as f:
            image = f["images"][idx_row["df_idx"]]
            mask = f["masks"][idx_row["df_idx"]]
            # mask does not need one hot encoding. it's only done for BinaryCrossEntropy or Dice

        if self.split != "all":
            image = image.transpose(1,2,0)
            image, mask = self.config.image_pre(image), self.config.target_pre(mask)
        if self.split in ("train", "trainvaltest"):
            image, mask = self.config.train_aug(image, mask)
        elif self.split in ("val", "test"):
            image, mask = self.config.eval_aug(image, mask)
        return image, mask, idx_row["df_idx"]
