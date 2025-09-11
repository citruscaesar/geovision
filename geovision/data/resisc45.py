from typing import Literal, Optional

import h5py
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import pandera.pandas as pa
import imageio.v3 as iio
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from torchvision.transforms import v2 as T

from geovision.data import Dataset, DatasetConfig
from geovision.io.local import FileSystemIO as fs

class Resisc45:
    local = Path.home() / "datasets" / "resisc45"
    class_names = (
        "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church", "circular_farmland", "cloud", 
        "commercial_area", "dense_residential", "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area", 
        "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park", "mountain", "overpass", "palace", "parking_lot", 
        "railway", "railway_station", "rectangular_farmland", "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", 
        "stadium", "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
    )
    #default_config = DatasetConfig()
    index_df_schema = pa.DataFrameSchema(
        columns={
            "image_path": pa.Column(str, coerce=True),
            "label_idx": pa.Column(int, coerce=True),
            "label_str": pa.Column(str, coerce=True),
        },
        index=pa.Index(int)
    )

    default_config = DatasetConfig(
        random_seed=42,
        tabular_sampler_name="stratified",
        tabular_sampler_params={"split_on": "label_str", "val_frac": 0.15, "test_frac": 0.15},
        image_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        target_pre=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=False)]),
        train_aug=T.Compose([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]),
        eval_aug=T.Identity()
    )

    @classmethod
    def load(
        cls, 
        table: Literal["index", "spatial", "spectral", "temporal"], 
        src: Literal["imagefolder", "hdf5"],
        subset: Literal["resisc45"] = "resisc45"
    ) -> pd.DataFrame:

        if src == "hdf5":
            return pd.read_hdf(cls.local / "hdf5" / f"{subset}.h5", key = table, mode = "r")

        elif src == "imagefolder":
            imagefolder_path = cls.local / "imagefolder"
            try:
                #raise OSError
                return pd.read_hdf(imagefolder_path / "metadata.h5", key = table, mode = "r")
            except OSError:
                assert table in ("index", "spatial")
                df = (
                    pd.DataFrame({"image_path": imagefolder_path.rglob("*.jpg")})
                    .assign(label_str = lambda df: df["image_path"].apply(lambda x: x.parent.name))
                    .assign(label_idx = lambda df: df["label_str"].apply(lambda x: cls.class_names.index(x)))
                    .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(Path(x.parent.name, x.name))))
                    .assign(image_width = 256)
                    .assign(image_height = 256)
                    .sort_values("image_path").reset_index(drop=True)
                )
                index_df = df[["image_path", "label_str", "label_idx"]]
                index_df.to_hdf(imagefolder_path / "metadata.h5", key = "index", mode = "w")

                spatial_df = df[["image_width", "image_height"]]
                spatial_df.to_hdf(imagefolder_path / "metadata.h5", key = "spatial", mode = "r+")

                if table == "index":
                    return index_df
                else: 
                    return spatial_df
    
    @classmethod
    def transform(
        cls,
        src: Literal["archive", "imagefolder"],
        to: Literal["hdf5"],
    ):
        if to == "hdf5" and src == "imagefolder":
            imagefolder_path = fs.get_valid_dir_err(cls.local / "imagefolder")
            hdf5_path = fs.get_new_dir(cls.local / "hdf5") / "resisc45.h5"

            index_df = cls.load("index", "imagefolder", "resisc45")
            spatial_df = cls.load("spatial", "imagefolder", "resisc45")

            with h5py.File(hdf5_path, mode = "w") as f:
                images = f.create_dataset("images", len(index_df), dtype = h5py.special_dtype(vlen=np.uint8))
                for idx, row in tqdm(index_df.iterrows(), total = len(index_df)):
                    image = iio.imread(imagefolder_path / row["image_path"], extension = ".jpg")
                    images[idx] = np.frombuffer(iio.imwrite("<bytes>", image, extension = ".jpg"), dtype = np.uint8)
            
            index_df.to_hdf(hdf5_path, key = "index", mode = "r+")
            spatial_df.to_hdf(hdf5_path, key = "spatial", mode = "r+")

class Resisc45_Imagefolder_Classification(Dataset):
    name = "resisc45"
    task = "classification"
    subtask = "multiclass"
    storage = "imagefolder"
    class_names = Resisc45.class_names 
    num_classes = 45 
    root = Resisc45.local / "imagefolder"
    schema = Resisc45.index_df_schema 
    config = Resisc45.default_config
    loader = Resisc45.load
    metadata_group_prefix = None

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=True)

    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, idx):
        idx_row = self.df.iloc[idx]
        image = iio.imread(idx_row["image_path"], extension=".jpg")
        image = self.config.image_pre(image)
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]

class Resisc45_HDF5_Classification(Dataset):
    name = "resisc45"
    task = "classification"
    subtask = "multiclass"
    storage = "hdf5"
    class_names = Resisc45.class_names 
    num_classes = 45 
    root = Resisc45.local / "hdf5" / "resisc45.h5"
    schema = Resisc45.index_df_schema 
    config = Resisc45.default_config
    loader = Resisc45.load
    metadata_group_prefix = None

    def __init__(self, split: Literal["train", "val", "test", "trainvaltest", "all"] = "all", config: Optional[DatasetConfig] = None):
        super().__init__(split, config)
        self.df = self.get_df(prefix_root_to_paths=False)

        # Load the entire dataset into memory ~ 6GB 
        self.images = np.empty(shape = (len(self.df), 256, 256, 3), dtype = np.uint8) 
        with h5py.File(self.root, mode = "r") as f:
            for idx, row in self.df.iterrows():
                self.images[idx] = iio.imread(BytesIO(f["images"][row["df_idx"]]), extension=".jpg")

    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, idx: int):
        idx_row = self.df.iloc[idx]
        image = self.config.image_pre(self.images[idx])
        if self.split in ("train", "trainvaltest"):
            image = self.config.train_aug(image)
        elif self.split in ("val", "test"):
            image = self.config.eval_aug(image)
        return image, idx_row["label_idx"], idx_row["df_idx"]