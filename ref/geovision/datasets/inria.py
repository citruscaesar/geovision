import os
import shutil
import h5py
import zipfile
from pathlib import Path
from pandas import DataFrame, concat
from torch import Tensor, float32
from numpy import uint8, eye, where, pad, dstack, clip, dsplit
from imageio.v3 import imread, imwrite
from litdata import optimize, StreamingDataset
from tqdm.auto import tqdm
from etl.etl import validate_dir
from etl.extract import extract_multivolume_archive
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform, Compose, ToImage, ToDtype, Identity
from torchvision.datasets.utils import download_url
from typing import Optional, Literal

class InriaSegmentation(Dataset):
    SUPERVISED_LOCATIONS = ("austin", "chicago", "kitsap", "vienna", "tyrol-w")
    UNSUPERVISED_LOCATIONS = ("bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e")
    URLS = ("https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005")
    DATASET_ARCHIVE_NAME = "NEW2-AerialImageDataset.zip"
    NUM_CLASSES = 2
    CLASS_NAMES = ("Background", "Foreground")
    NAME = "urban_footprint"
    TASK = "segmentation"
    EYE = eye(NUM_CLASSES, dtype = "float")
    DEFAULT_IMAGE_TRANSFORM = Compose([ToImage(), ToDtype(float32, scale=True)])
    DEFAULT_TARGET_TRANSFORM = Compose([ToImage(), ToDtype(float32, scale=False)])
    DEFAULT_COMMON_TRANSFORM = Identity()
    SCENE_SHAPE = (5000, 5000, 3)

    @classmethod
    def supervised_df(cls, val_split: float = 0.2, test_split: float = 0.2, random_seed: int = 42, **kwargs) -> DataFrame:
        sup_file_loc_pairs = [(f"{x}{num}.tif", x) for x in cls.SUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"file": sup_file_loc_pairs})
                .assign(scene_idx = lambda df: df.index)
                .assign(scene_name = lambda df: df.file.apply(
                    lambda x: x[0]))
                .assign(loc = lambda df: df.file.apply(
                    lambda x: x[1]))
                .drop(columns = "file")
                .pipe(cls.__assign_train_test_val_splits, val_split, test_split, random_seed)) # type: ignore

    @classmethod
    def unsupervised_df(cls) -> DataFrame:
        unsup_files = [f"{x}{num}.tif" for x in cls.UNSUPERVISED_LOCATIONS for num in range(1, 37)]
        return (DataFrame({"scene_name": unsup_files})
                .assign(scene_idx = lambda df: df.index)
                .assign(split = "unsup")
                [["scene_idx", "scene_name", "split"]])

    @classmethod
    def scene_df(cls, val_split: float = 0.2, test_split: float = 0.2, random_seed: int = 42, **kwargs) -> DataFrame:
        r"""
        Parameters
        ----------
        random_seed: int 
            used to randomly sample dataset to generate train-val-test splits
        test_split: float, between [0, 1]
            proportion of dataset to use as test data
        val_split: float, between [0, 1]
            proportion of dataset to use as validation data
        **kwargs: dict[str, Any]
            for when user feels lazy
        """
        return (
            concat([cls.supervised_df(val_split, test_split, random_seed), cls.unsupervised_df()])
            .assign(hbeg = 0).assign(hend = 5000).assign(wbeg = 0).assign(wend = 5000)
            .assign(tile_name = lambda df: df.apply(
                lambda x: f"{x['scene_name'].removesuffix('.tif')}_{x['hbeg']}_{x['hend']}_{x['wbeg']}_{x['wend']}.tif",
                axis = 1))
            [["scene_idx", "scene_name", "tile_name", "split", "hbeg", "hend", "wbeg", "wend"]]
            .reset_index(drop = True)
        )
                                     
    @classmethod
    def tiled_df(
            cls, 
            random_seed: int,
            test_split: float,
            val_split: float,
            tile_size: tuple[int, int],
            tile_stride: tuple[int, int],
            **kwargs) -> DataFrame:
        r"""
        Parameters
        ----------
        random_seed: int 
            used to randomly sample dataset to generate train-val-test splits
        test_split: float, between [0, 1]
            proportion of dataset to use as test data
        val_split: float, between [0, 1]
            proportion of dataset to use as validation data
        tile_size: tuple[int, int]
            size (x, y) of the sliding window (kernel) used to draw samples 
        tile_stride: tuple[int, int] 
            stride (x, y) of the sliding window (kernel) used to draw samples 
        **kwargs: dict[str, Any]
            for when user feels lazy
        """
        assert isinstance(tile_size, tuple) and len(tile_size) == 2, "Invalid Tile Size"
        assert isinstance(tile_stride, tuple) and len(tile_stride) == 2, "Invalid Tile Stride"

        df = cls.scene_df(val_split, test_split, random_seed)
        assert {"scene_idx", "scene_name", "split"}.issubset(df.columns), f"scene_df missing columns"

        tile_dfs = list()
        for _, row in df.iterrows():
            table: dict[str, list] = {
                "tile_name": list(),
                "scene_name": list(),
                "hbeg": list(),
                "hend": list(),
                "wbeg": list(),
                "wend": list()
            }

            scene_name = Path(row["scene_name"])
            for x in range(0, cls.__num_windows(cls.SCENE_SHAPE[0], tile_size[0], tile_stride[0])):
                for y in range(0, cls.__num_windows(cls.SCENE_SHAPE[1], tile_size[1], tile_stride[1])):
                    hbeg = x*tile_stride[0]
                    hend = x*tile_stride[0]+tile_size[0]
                    wbeg = y*tile_stride[1]
                    wend = y*tile_stride[1]+tile_size[1]
                    name = f"{scene_name.stem}_{hbeg}_{hend}_{wbeg}_{wend}{scene_name.suffix}"

                    table["tile_name"].append(name)
                    table["scene_name"].append(scene_name.name)
                    table["hbeg"].append(hbeg)
                    table["hend"].append(hend)
                    table["wbeg"].append(wbeg)
                    table["wend"].append(wend)

            tile_dfs.append(
                DataFrame(table)
                .assign(scene_idx = row["scene_idx"])
                .assign(split = row["split"]))

        return (
            concat(tile_dfs)
            .reset_index(drop = True)
            [["scene_idx", "scene_name", "tile_name", "split", "hbeg", "hend", "wbeg", "wend"]]
        )

    @classmethod
    def show_tiles_along_one_dim(cls, dim_len: int, kernel: int, stride: int, padding: Optional[int] = None) -> None:
        PADDING = cls.__padding_required(dim_len, kernel, stride) if padding is None else padding
        NUM_TILES = cls.__num_windows(dim_len, kernel, stride) 

        print(f"Can Create {NUM_TILES} Windows")
        print(f"If Image Is Padded by: {PADDING} Pixels\n")

        for tile_idx in range(0, NUM_TILES):
            print(f"Tile #{tile_idx} -> [{tile_idx * stride}:{(tile_idx * stride + kernel)})")

    @classmethod
    def download(cls, root: Path, low_storage: bool):
        downloads = root / "downloads"
        downloads.mkdir(exist_ok=True)

        print(f"Downloading .7z archives to {downloads}")
        for url in tqdm(cls.URLS):
            download_url(url, str(downloads))

        dataset_archive_path = (root / cls.DATASET_ARCHIVE_NAME)
        print(f"Extracting dataset archive to {dataset_archive_path}")
        extract_multivolume_archive(downloads / "aerialimagelabeling.7z", root)
        if low_storage:
            print(f"Deleting downloaded .7z archives from {downloads}")
            shutil.rmtree(str(downloads))

    @staticmethod
    def read_image(src_uri: str, hbeg:int, hend:int, wbeg:int, wend:int):
        image = imread(src_uri, extension = ".tif")
        H, W = image.shape[0], image.shape[1]
        image = image[hbeg: min(hend, H), wbeg: min(wend, W)].copy()
        if hend > H or wend > W:
            if image.ndim == 2: 
                image = pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W))), "constant", constant_values = 0).copy()
            else:
                image = pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
        return image

    @classmethod
    def write_to_files(cls, root: Path, target: Path, df: DataFrame, **kwargs):
        r"""
        Parameters
        ----------
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located

        target: Path
            Directory to write prepared files to, additional subdirectories are
            created for image, mask and unsupervised tiles 

        df: DataFrame
            DataFrame with appropriate columns containing train-val-test splits
            df.columns = {scene_name, tile_name, split, hbeg, hend, wbeg, wend}

        **kwargs: dict[str, Any], optional
            for being lazy 
        """
        DATASET_ZIP = root / "NEW2-AerialImageDataset.zip"
        assert DATASET_ZIP.is_file(), f"Dataset Archive Missing @ [{DATASET_ZIP}]"
        IMAGES = validate_dir(target, "images")
        MASKS = validate_dir(target, "masks")
        UNSUP = validate_dir(target, "unsupervised")

        for _, row in tqdm(df.iterrows(), desc = "Writing to Files", total = len(df)):
            _crop_dims = (row["hbeg"], row["hend"], row["wbeg"], row["wend"])
            if row["split"] == "unsup":
                imwrite(
                    uri = str(UNSUP/row["tile_name"]),
                    image = cls.read_image(
                        (DATASET_ZIP/"AerialImageDataset"/"test"/"images"/row["scene_name"]).as_posix(), *_crop_dims),
                    extension = ".tif"
                )
            else:
                imwrite(
                    uri = str(IMAGES/row["tile_name"]),
                    image = cls.read_image(
                        (DATASET_ZIP/"AerialImageDataset"/"train"/"images"/row["scene_name"]).as_posix(), *_crop_dims),
                    extension = ".tif"
                )
                imwrite(
                    uri = str(MASKS/row["tile_name"]),
                    image = cls.read_image(
                        (DATASET_ZIP/"AerialImageDataset"/"train"/"gt"/row["scene_name"]).as_posix(), *_crop_dims),
                    extension = ".tif"
                )
        df.to_csv(target/"dataset.csv", index = False)

    @classmethod
    def write_to_hdf(cls, root: Path, target: Path, df: DataFrame) -> None:
        r"""
        Parameters
        ----------
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located

        target: Path
            Directory to write prepared hdf5 file to 

        df: DataFrame
            DataFrame with appropriate columns containing train-val-test splits
            df.columns = {scene_name, tile_name, split, hbeg, hend, wbeg, wend}

        **kwargs: dict[str, Any], optional
            to accomodate laziness
        """

        # bits * numimages * width * height * num_bands(RGB+Mask)
        # size_in_bits = 8 * 180 * 5000 * 5000 * (3+3+2) 
        # size_in_gigabytes = size_in_bits / (8 * 1024 * 1024 * 1024)
        # ~33.52GB
        DATASET_ZIP = root / "NEW2-AerialImageDataset.zip"
        assert DATASET_ZIP.is_file(), f"Dataset Archive Missing @ [{DATASET_ZIP}]"
        target = validate_dir(target)
        HDF_DATASET = target / "inria.h5"

        def get_dataset_dims(split_df: DataFrame) -> tuple[int, int, int, int]:
            display(split_df)
            row = split_df.iloc[0]
            N = len(split_df)
            H = row["hend"] - row["hbeg"]
            W = row["wend"] - row["wbeg"]
            B = 3 if split == "unsup" else 5
            #print(f"{split} dataset dims: {N, H, W, B}")
            return N, H, W, B
        
        with h5py.File(HDF_DATASET, 'w') as f:
            for split in tqdm(df["split"].unique(), desc = "Splits", position=0):
                split_df = df[df["split"] == split].reset_index(drop = True)
                split_ds = f.create_dataset(split, get_dataset_dims(split_df), uint8)
                for idx, row in tqdm(split_df.iterrows(), desc = "Images Written", position=1, leave = False, total = len(split_df)):
                    _crop_dims = (row["hbeg"], row["hend"], row["wbeg"], row["wend"])
                    if split == "unsup":
                        split_ds[idx] = cls.read_image(
                        (DATASET_ZIP/"AerialImageDataset"/"test"/"images"/row["scene_name"]).as_posix(), *_crop_dims),
                    else:    
                        image = cls.read_image(
                            (DATASET_ZIP/"AerialImageDataset"/"train"/"images"/row["scene_name"]).as_posix(), *_crop_dims)
                        mask = cls.read_image(
                            (DATASET_ZIP/"AerialImageDataset"/"train"/"gt"/row["scene_name"]).as_posix(), *_crop_dims)
                        mask = eye(2, dtype = uint8)[clip(mask, 0, 1)]
                        split_ds[idx] = dstack([image, mask])

    @classmethod
    def write_to_litdata(cls, root: Path, shards: Path, df: Optional[DataFrame] = None, **kwargs):
        """
        Parameters
        -----
        root: Path
            Directory where "NEW2-AerialImageDataset.zip" is located
        shards: Path
            Parent directory to store prepared shards, usually Path.home() / "shards" / "dataset_name", 
            shards are stored in subdirectories within this directory
        df: DataFrame, optional
            DataFrame containing scene_idx, scene_name and split, used for saving train-val-test-unsup-splits, 
            if not provided, it is generated randomly using **kwargs
        **kwargs: dict
            random_seed: int
                for reproducibility
            test_split: float[0, 1]
                proportion of test samples
            val_split: float[0, 1]
                proportion of validation samples
            num_workers: int
                number of processes used to encode dataset, defaults to os.cpu_count() 
            shard_size_in_mb: int 
                size of each chunk in megabytes, defaults to 512 
        """

        EYE = eye(2, dtype = uint8)
        DATASET = root / cls.DATASET_ARCHIVE_NAME / "AerialImageDataset"
        assert DATASET.parent.is_file(), f"Dataset Archive Missing @ [{DATASET.parent}]"

        def encode_supervised_samples(input):
            _, row = input
            image = imread(DATASET/"train"/"images"/row["scene_name"])
            mask =  imread(DATASET/"train"/"gt"/row["scene_name"])
            mask = EYE[where(mask.squeeze() == 255, 1, 0).astype(uint8)]
            return {
                "scene_idx": row["scene_idx"], 
                "image": image,
                "mask": mask,
            } 

        def encode_unsupervised_samples(input):
            _, row = input
            image = imread(DATASET/"test"/"images"/row["scene_name"])
            return {
                "scene_idx": row["scene_idx"], 
                "image": image,
            } 
        
        # TODO: Take tile_size and tile_stride as input and add an option to make tiled dataset
        scene_df = df if df is not None else cls.scene_df(**kwargs)
        for split in ("train", "val", "test", "unsup"):
            optimize(
                fn = encode_unsupervised_samples if split=="unsup" else encode_supervised_samples,
                inputs = list(scene_df.pipe(cls._subset_df, split).iterrows()),
                output_dir = str(shards / split),
                num_workers = kwargs.get("num_workers", os.cpu_count()),
                chunk_bytes = kwargs.get("shard_size_in_mb", 256) * 1024 * 1024,
            )

    @staticmethod
    def __assign_train_test_val_splits(df: DataFrame, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        test = (df
                .groupby("loc", group_keys=False)
                .apply(
                    lambda x: x.sample(
                    frac = test_split,
                    random_state = random_seed,
                    axis = 0)
                .assign(split = "test")))
        val = (df
                .drop(test.index, axis = 0)
                .groupby("loc", group_keys=False)
                .apply(
                    lambda x: x.sample( 
                    frac = val_split / (1-test_split),
                    random_state = random_seed,
                    axis = 0)
                .assign(split = "val")))
        train = (df
                  .drop(test.index, axis = 0)
                  .drop(val.index, axis = 0)
                  .assign(split = "train"))

        return (concat([train, val, test])
                    .sort_index()
                    .drop("loc", axis = 1))

    @staticmethod
    def __num_windows(length: int, kernel: int, stride: int) -> int:
            return (length - kernel - 1) // stride + 2

    @staticmethod
    def __padding_required(length: int, kernel: int, stride: int) -> int:
        num_windows = (length - kernel - 1) // stride + 2
        return (num_windows - 1) * stride + kernel - length 

    @staticmethod
    def __extract_image_to_dst(src_path: str, dst_path: Path, zip_file_obj: zipfile.ZipFile):
        with open(dst_path, "wb") as dst_file_obj:
            with zip_file_obj.open(src_path, "r") as src_file_obj:
                shutil.copyfileobj(src_file_obj, dst_file_obj)

    #@staticmethod
    #def _subset_df(df: DataFrame, split: str):
        #if split == "all":
            #return df[df["split"] != "unsup"].reset_index(drop=True)
        #elif split == "trainval":
            #return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        #return (df[df.split == split].reset_index(drop=True))

    @staticmethod
    def _subset_df(df: DataFrame, split: str) -> DataFrame:
        train_split = df[df["split"] == "train"].reset_index(drop = True)
        val_split = df[df["split"] == "val"].reset_index(drop = True)
        test_split = df[df["split"] == "test"].reset_index(drop = True)
        unsup_split = df[df["split"] == "unsup"].reset_index(drop = True)
        if split == "train":
            return train_split
        elif split == "val":
            return val_split
        elif split == "test":
            return test_split
        elif split == "unsup":
            return unsup_split
        elif split == "trainval":
            return concat([train_split, val_split], axis = 0)
        elif split == "all":
            return concat([train_split, val_split, test_split, unsup_split], axis = 0)

    @staticmethod
    def _prefix_root_to_df(df: DataFrame, root: Path) -> DataFrame:
        return (
            df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(root/x)))
            .assign(mask_path = lambda df: df["mask_path"].apply(lambda x: str(root/x)))
        ) 

class InriaImageFolder(InriaSegmentation):
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
        ) -> None:

        root = root / "scenes"

        assert split in ("train", "val", "test", "trainval", "unsup", "all"), "Invalid Split"
        self.root = root
        self.split = split
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
            "tile_size" : tile_size,
            "tile_stride" : tile_stride
        }

        if isinstance(df, DataFrame):
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.tiled_df(**experiment_kwargs)
        else:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.scene_df(**experiment_kwargs)

        assert {"scene_name", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(self.df.columns), "incorrect dataframe schema"
        self.df = (
            self.df
            .assign(image_path = lambda df: df.apply(lambda x: str(Path("images", x["scene_name"])) if x["split"] != "unup" else str(Path("unsup", x["scene_name"])), axis = 1))
            .assign(mask_path = lambda df: df["image_path"].apply(lambda x: str(Path(str(x).replace("image", "mask")))))
            .assign(df_idx = lambda df: df.index)
        )
        self.split_df  = (
            self.df
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, root)
        )
    
    def __len__(self):
        return len(self.split_df)
    
# NumPy slicing creates a view instead of a copy as in the case of built-in Python sequences such as string, tuple and list.
# Care must be taken when extracting a small portion from a large array which becomes useless after the extraction,
# because the small portion extracted contains a reference to the large original array whose memory will not be released
# until all arrays derived from it are garbage-collected.
# In such cases an explicit copy() is recommended.

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        row = self.split_df.iloc[idx]
        H, W, hbeg, hend, wbeg, wend = self.SCENE_SHAPE[0], self.SCENE_SHAPE[1], row["hbeg"], row["hend"], row["wbeg"], row["wend"]

        image_scene = imread(row["image_path"])
        image = image_scene[hbeg: min(hend, H), wbeg: min(wend, W), :].copy()
        del image_scene

        mask_scene = imread(row["mask_path"])
        mask = mask_scene[hbeg: min(hend, H), wbeg: min(wend, W)].copy()
        del mask_scene

        if hend > H or wend > W:
            image = pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0)
            mask = pad(mask, ((0, max(0, hend - H)), (0, max(0, wend - W))), "constant", constant_values = 0)

        mask = self.EYE[where(mask == mask.max(), 1, 0)]
        image = self.image_transform(image)
        mask = self.target_transform(mask)

        if self.split == "train":
            image, mask = self.common_transform([image, mask])
        return image, mask, row["df_idx"]

#class InriaHDF5(InriaSegmentation):
    #def __init__(
            #self,
            #root: Path,
            #df: Optional[DataFrame] = None,
            #split: str = "train",
            #test_split: float = 0.2,
            #val_split: float = 0.2,
            #random_seed: int = 42,
            #tile_size: Optional[tuple[int, int]] = None,
            #tile_stride: Optional[tuple[int, int]] = None,
            #image_transform: Optional[Transform] = None,
            #target_transform: Optional[Transform] = None,
            #common_transform: Optional[Transform] = None,
            #**kwargs,
        #) -> None:
        #assert root.is_file() and (root.suffix == ".h5" or root.suffix == ".hdf5"), f"{root} does not point to an .h5/.hdf5 file"
        #assert split in ("train", "val", "test", "unsup", "trainval", "all"), f"provided split [{split}] is invalid"
        #self.root = root
        #self.split = split
        #self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        #self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        #self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        #experiment_kwargs = {
            #"random_seed" : random_seed,
            #"val_split": val_split,
            #"test_split": test_split,
            #"tile_size" : tile_size,
            #"tile_stride" : tile_stride
        #}

        #if isinstance(df, DataFrame):
            #print(f"{split} custom dataset @ [{self.root}]")
            #self.df = df
        #elif tile_size is not None and tile_stride is not None:
            #print(f"{split} tiled dataset @ [{self.root}]")
            #self.df = self.tiled_df(**experiment_kwargs)
        #else:
            #print(f"{split} scene dataset @ [{self.root}]")
            #self.df = self.scene_df(**experiment_kwargs)

        #assert {"scene_idx", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(self.df.columns), "incorrect dataframe schema"
        #self.df = self.df.assign(df_idx = lambda df: df.index)
        #self.split_df = self.df.pipe(self._subset_df, split)    

    #def __len__(self):
        #return len(self.split_df)
    
    #def __getitem__(self, idx):
        #row = self.split_df.iloc[idx]
        #with h5py.File(self.root, mode = "r") as f:
            #image_mask = f[row["split"]][idx]
        #image = self.image_transform(image_mask[:, :, :3].copy())
        #mask = self.target_transform(image_mask[:, :, 3:].copy())
        #del image_mask
        #if self.split == "train":
            #image, mask = self.common_transform([image, mask])
        #return image, mask, row["df_idx"]

    #def _readimage(self, dataset_name: Literal["supervised", "unsupervised"], idx: int, H:int, W:int, hbeg: int, hend: int, wbeg: int, wend: int):
        #with h5py.File(self.root, mode = "r") as f:
            #image = f[dataset_name][idx, hbeg: min(hend, H), wbeg: min(wend, W)]
            #if hend > H or wend > W:
                    #return pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0)
            #return image 

class InriaHDF5(InriaSegmentation):
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: str = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            tile_size: Optional[tuple[int, int]] = None,
            tile_stride: Optional[tuple[int, int]] = None,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs,
        ) -> None:
        assert root.is_file() and (root.suffix == ".h5" or root.suffix == ".hdf5"), f"{root} does not point to an .h5/.hdf5 file"
        assert split in ("train", "val", "test", "unsup", "trainval", "all"), f"provided split [{split}] is invalid"
        self.root = root
        self.split = split
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM 

        experiment_kwargs = {
            "random_seed" : random_seed,
            "val_split": val_split,
            "test_split": test_split,
            "tile_size" : tile_size,
            "tile_stride" : tile_stride
        }

        if isinstance(df, DataFrame):
            print(f"{split} custom dataset @ [{self.root}]")
            self.df = df
        elif tile_size is not None and tile_stride is not None:
            print(f"{split} tiled dataset @ [{self.root}]")
            self.df = self.tiled_df(**experiment_kwargs)
        else:
            print(f"{split} scene dataset @ [{self.root}]")
            self.df = self.scene_df(**experiment_kwargs)

        assert {"scene_idx", "split", "hbeg", "hend", "wbeg", "wend"}.issubset(self.df.columns), "incorrect dataframe schema"
        self.df = self.df.assign(df_idx = lambda df: df.index)
        self.split_df = self.df.pipe(self._subset_df, split)    

    def __len__(self):
        return len(self.split_df)
        
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        H, W = self.SCENE_SHAPE[0], self.SCENE_SHAPE[1]
        hbeg, hend, wbeg, wend = row["hbeg"], row["hend"], row["wbeg"], row["wend"]
        with h5py.File(self.root, mode = "r") as f:
            image_mask = f["supervised"][row["scene_idx"], hbeg: min(hend, H), wbeg: min(wend, W)] 
        image = image_mask[:, :, :3].copy()
        mask = image_mask[:, :, 3:].copy()
        del image_mask
        if hend > H or wend > W: 
            image = pad(image, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
            mask = pad(mask, ((0, max(0, hend - H)), (0, max(0, wend - W)), (0, 0)), "constant", constant_values = 0).copy()
        image, mask = self.image_transform(image), self.target_transform(mask)
        if self.split == "train":
            image, mask = self.common_transform([image, mask])
        return image, mask, row["df_idx"]



class InriaLitData(StreamingDataset, InriaSegmentation):
    def __init__(
            self,
            root: Path | str,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "test"] = "train",
            test_split: float = 0.2,
            val_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            shuffle: bool = False,
            cache_limit: int | str = "30GB",
            **kwargs,
    ) -> None:
        assert split in ("train", "val", "test"), "Invalid Split"
        root = str(root/split) if isinstance(root, Path) else f"{root.removesuffix('/')}/{split}/"
        print(f"{split} scene dataset @ [{root}]")

        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM 
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM
        self.df = (
            df if isinstance(df, DataFrame) else self.scene_df(
                random_seed = random_seed,
                test_spilt = test_split,
                val_split = val_split
            )
            .assign(df_idx = lambda df: df.index)
        )

        super().__init__(
            input_dir = root,
            shuffle = shuffle,
            seed = random_seed,
            max_cache_size = cache_limit,
        )
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        scene = super().__getitem__(idx)
        df_idx = self.df.iloc[idx.index]["df_idx"]
        return (*self.common_transform([
                    self.image_transform(scene["image"]), 
                    self.target_transform(scene["mask"])]), 
                df_idx)