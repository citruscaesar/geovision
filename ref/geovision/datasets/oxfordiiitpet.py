from pathlib import Path
from numpy import eye
from torch import Tensor, float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Identity
)
from torchvision.datasets.utils import download_and_extract_archive

from typing import Any, Literal, Optional
from numpy.typing import NDArray
from torchvision.transforms.v2 import Transform

class OxfordIIITPetBase:
    NUM_IMAGES_PER_CLASS = 200
    NUM_SEGMENTATION_CLASSES = 3
    BAD_IMAGES = {"Egyptian_Mau_14", 
                  "Egyptian_Mau_129",
                  "Egyptian_Mau_186",
                  "staffordshire_bull_terrier_2",
                  "staffordshire_bull_terrier_22",
                  "Abyssinian_5"}
    RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    DEFAULT_IMAGE_TRANSFORM = Compose([ToImage(), ToDtype(float32, scale=True)])
    DEFAULT_TARGET_TRANSFORM = Compose([ToImage(), ToDtype(int64, scale=False)])
    DEFAULT_COMMON_TRANSFORM = Identity()

    @classmethod
    def download_and_extract(cls, root: Path) -> None:
        """
        Downloads Oxford IIIT Pet Dataset and Sets Root to Correct Directory\n
        Root is reset because of how the torchvision utility download_and_extract_archive works.
        """
        # TODO: Use torchvision download_url and custom extract function 
        for url, md5 in cls.RESOURCES:
                download_and_extract_archive(url, download_root = root.parent.as_posix(), md5=md5)
        root = root.parent / "oxford-iiit-pet"


    @classmethod
    def dataset_df(cls, root: Path, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        return(
            DataFrame({"image_path": (root/"images").rglob("*.jpg")})
            .assign(image_path = lambda df: df["image_path"].apply(
                lambda x: Path(x.parent.stem, x.name)))
            .assign(mask_path = lambda df: df["image_path"].apply(
                lambda x: Path("annotations", "trimaps", f"{x.stem}.png")))
            .assign(label_str = lambda df: df["image_path"].apply(
                lambda x: x.stem.split("_")[0].lower()))
            .pipe(cls._drop_bad_images, cls.BAD_IMAGES)
            .pipe(cls._assign_train_test_val_splits, val_split, test_split, random_seed))    
    
    @staticmethod
    def _drop_bad_images(df: DataFrame, bad_images: set[str]) -> DataFrame:
        return df[~df.image_path.apply(lambda x: x.stem in bad_images)]
    
    @staticmethod 
    def _assign_train_test_val_splits(df: DataFrame, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        test = (df.groupby("label_str", group_keys=False)
                  .apply(lambda x: x.sample(frac = test_split, random_state = random_seed, axis = 0)
                  .assign(split = "test")))
        val = (df.drop(test.index, axis = 0)
                 .groupby("label_str", group_keys=False)
                 .apply(lambda x: x.sample(frac = val_split / (1-test_split), random_state = random_seed, axis = 0)
                 .assign(split = "val")))
        train = (df.drop(test.index, axis = 0)
                   .drop(val.index, axis = 0)
                   .assign(split = "train"))
        return concat([train, val, test]).sort_values("image_path").reset_index(drop = True)

    @staticmethod
    def _subset_df(df: DataFrame, split: str) -> DataFrame:
        if split == "trainval":
            return (df[(df["split"] == "train") | (df["split"] == "val")].reset_index(drop=True)) # type: ignore
        return (df[df["split"] == split].reset_index(drop=True))

    @staticmethod
    def _prefix_root_to_df(df: DataFrame, root: Path) -> DataFrame:
        df["image_path"] = df["image_path"].apply(lambda x: str(root/x))
        if "mask_path" in df.columns:
            df["mask_path"] = df["mask_path"].apply(lambda x: str(root/x))
        return df
    
class OxfordIIITPetSegmentation(OxfordIIITPetBase):
    CLASS_NAMES = ("foreground", "background", "outline")
    NUM_CLASSES = 3
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test"] = "train",
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download = False,
            **kwargs
        ) -> None:
        assert split in ("train", "val", "trainval", "test"), "Invalid Split"

        self.root = root
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.target_transform = target_transform or self.DEFAULT_TARGET_TRANSFORM
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM
        _reproducibility_kwargs = {
            "random_seed": random_seed,
            "val_split": val_split,
            "test_split": test_split
        } 

        self.df = df if isinstance(df, DataFrame) else self.segmentation_df(self.root, **_reproducibility_kwargs)
        self.split_df = (
            self.df
            .assign(df_idx = lambda df: df.index)
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, self.root)
        )
        self.identity_matrix = eye(self.NUM_SEGMENTATION_CLASSES)

    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor, int]:
        row_idx = self.split_df.iloc[idx]
        image = imread(row_idx["image_path"]).squeeze() # type: ignore
        mask = imread(row_idx["mask_path"]).squeeze()
        mask = self.identity_matrix[mask-1]
        return *self.common_transform([self.image_transform(image), self.target_transform(mask)]), row_idx["df_idx"]

    @classmethod
    def segmentation_df(cls, root: Path, val_split:float, test_split:float, random_seed: int) -> DataFrame:
        return (cls.dataset_df(root, val_split, test_split, random_seed)
                   .drop(columns="label_str"))
            
class OxfordIIITPetClassification(OxfordIIITPetBase):
    CLASS_NAMES = ('abyssinian', 'american', 'basset', 'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british', 'chihuahua', 'egyptian', 'english', 'german', 'great', 'havanese', 'japanese', 'keeshond', 'leonberger', 'maine', 'miniature', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll', 'russian', 'saint', 'samoyed', 'scottish', 'shiba', 'siamese', 'sphynx', 'staffordshire', 'wheaten', 'yorkshire')
    NUM_CLASSES = 35
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test"] = "train",
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs
    ) -> None:
        assert split in ("train", "val", "trainval", "test"), "Invalid Split"

        self.root = root
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

        _reproducibility_kwargs = {
            "random_seed": random_seed,
            "val_split": val_split,
            "test_split": test_split
        }    
        self.df = df if isinstance(df, DataFrame) else self.classification_df(self.root, **_reproducibility_kwargs)

        self.split_df = (
            self.df
            .assign(df_idx = lambda df: df.index)
            .pipe(self._subset_df, split)
            .pipe(self._prefix_root_to_df, self.root)
        )

    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx) -> tuple[Tensor, int, int]:
        row_idx = self.df.iloc[idx]
        image = imread(row_idx["image_path"]).squeeze()
        return self.common_transform(self.image_transform(image)), row_idx["label"], row_idx["df_idx"] 

    @classmethod 
    def classification_df(cls, root: Path, val_split: float, test_split: float, random_seed: int) -> DataFrame:
        return (cls.dataset_df(root, val_split, test_split, random_seed)
                   .drop(columns = ["mask_path"])
                   .pipe(cls.__encode_labels, cls.CLASS_NAMES))

    @staticmethod 
    def __encode_labels(df: DataFrame, class_names: tuple[str, ...]) -> DataFrame:
        return df.assign(label_idx = lambda df: df["label_str"].apply(
            lambda x: class_names.index(x)))