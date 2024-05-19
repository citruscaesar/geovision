import shutil
from pathlib import Path
from numpy import isin, stack
from torch import tensor, float32, int64
from pandas import DataFrame, concat
from imageio.v3 import imread
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms.v2 import (
    Compose, ToImage, ToDtype, Resize, Identity 
)
from torchvision.datasets.utils import download_and_extract_archive

from typing import Any, Literal, Optional
from torch.utils.data import Dataset
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class ImagenetteClassification(Dataset):
    URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

    CLASS_LABELS = {
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n02102040': 'english_springer',
        'n03394916': 'french_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute',
        'n01440764': 'tench',
    }

    NAME = "imagenette"
    TASK = "classification"
    NUM_CLASSES = len(CLASS_LABELS) 
    CLASS_NAMES = tuple(sorted(CLASS_LABELS.values()))

    MEANS = [0.485, 0.456, 0.406]
    STD_DEVS = [0.229, 0.224, 0.225]

    DEFAULT_IMAGE_TRANSFORM = Compose([
        ToImage(),
        ToDtype(float32, scale = True),
        Resize((224, 224), antialias=True)
    ]) 

    DEFAULT_COMMON_TRANSFORM = Identity() 

    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            download = False,
            **kwargs
        ) -> None:
        self.root = root
        if download:
            # TODO: use download_url and write custom extraction function to avoid moving the dataset 
            download_and_extract_archive(self.URL, str(root))
            self.__move_dataset_up_one_level()
        assert self.root.is_dir(), "Root Does Not Exist"
        assert split in ("train", "val", "trainval", "test", "all"), "invalid split"
        self.split = split 
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

        print(f"{self.split} dataset at {self.root}")

        reproducibility_kwargs = {
            "val_split" : val_split,
            "test_split" : test_split,
            "random_seed" : random_seed
        }
        self.df = df if isinstance(df, DataFrame) else self.classification_df(self.root, **reproducibility_kwargs)
        self.split_df = (
            self.df
            .assign(df_idx = lambda df: df.index)
            .pipe(self.__subset_df)
            .pipe(self.__prefix_root_to_df)
        )
        self.df = self.df.assign(image_path = lambda df: df["image_path"].apply(lambda x: str(x)))

    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, int, int]:
        idx_row = self.split_df.iloc[idx]
        image = imread(idx_row["image_path"]).squeeze()
        image = self.__stacked_grayscale_image(image) if image.ndim == 2 else image
        image = self.common_transform(self.image_transform(image))

        return image, idx_row["label_idx"], idx_row["df_idx"]  
        
    def __stacked_grayscale_image(self, image: NDArray) -> NDArray:
            return stack((image,)*3, axis = -1)

    def __subset_df(self, df: DataFrame) -> DataFrame:
        if self.split == "all":
            return df
        elif self.split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == self.split].reset_index(drop=True))

    def __prefix_root_to_df(self, df: DataFrame) -> DataFrame:
        return df.assign(image_path = lambda df: df["image_path"].apply(lambda x: str(self.root/x)))
    
    def __move_dataset_up_one_level(self) -> None:
        print(f"Copying extracted files to: {self.root}")
        shutil.copytree(str(self.root / "imagenette2"), str(self.root), dirs_exist_ok=True)
        print(f"Cleaning up duplicates")
        shutil.rmtree(str(self.root / "imagenette2"))

    @classmethod
    def classification_df(cls, root: Path, val_split, test_split, random_seed) -> DataFrame:
        df = (DataFrame({"image_path": list(root.rglob("*.JPEG"))})
              .assign(image_path = lambda df: df.image_path.apply(
                  lambda x: Path(x.parents[1].stem, x.parents[0].stem, x.name)))
              .assign(label_str = lambda df: df.image_path.apply(
                  lambda x: cls.CLASS_LABELS[x.parent.stem].lower().replace(' ', '_')))
              .pipe(cls._encode_label, cls.CLASS_NAMES)
              .assign(split = lambda df: df.image_path.apply(
                  lambda x: "test" if x.parents[1].stem == "val" else "train"))
              .pipe(cls._assign_train_test_val_splits, val_split, test_split, random_seed)
        )
        return df

    @staticmethod 
    def _encode_label(df: DataFrame, class_names: tuple) -> DataFrame:
        return df.assign(label_idx = lambda df: df["label_str"].apply(
            lambda x: class_names.index(x))) # type: ignore

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

        return concat([train, val, test]).reset_index(drop = True)
