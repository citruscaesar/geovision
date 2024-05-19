from pathlib import Path
from torch import float32
from pandas import DataFrame, concat
from imageio.v3 import imread
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Identity

from typing import Any, Literal, Optional
from torch.utils.data import Dataset
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.v2 import Transform

class PlantDiseaseClassification(Dataset):
    DEFAULT_IMAGE_TRANSFORM = Compose([
        ToImage(),
        ToDtype(float32, scale = True)
    ])

    DEFAULT_COMMON_TRANSFORM = Compose([
        Identity(),
    ])

    NAME = "plant_village"
    TASK = "classification"

    CLASS_NAMES = (
        'apple-apple_scab',
        'apple-black_rot',
        'apple-cedar_apple_rust',
        'apple-healthy',
        'banana-bbs',
        'banana-bbw',
        'banana-healthy',
        'blueberry-healthy',
        'cherry_(including_sour)-healthy',
        'cherry_(including_sour)-powdery_mildew',
        'corn_(maize)-cercospora_leaf_spot gray_leaf_spot',
        'corn_(maize)-common_rust_',
        'corn_(maize)-healthy',
        'corn_(maize)-northern_leaf_blight',
        'grape-black_rot',
        'grape-esca_(black_measles)',
        'grape-healthy',
        'grape-leaf_blight_(isariopsis_leaf_spot)',
        'groundhut-healthy',
        'groundnut-deficient',
        'groundnut-early_leaf_spot',
        'groundnut-early_rust',
        'groundnut-late_leaf_spot',
        'groundnut-rust',
        'orange-haunglongbing_(citrus_greening)',
        'paddy-blight',
        'paddy-healthy',
        'paddy-smut',
        'paddy_brownspot-paddy_brownspot',
        'peach-bacterial_spot',
        'peach-healthy',
        'pepper_bell-bacterial_spot',
        'pepper_bell-healthy',
        'potato-early_blight',
        'potato-healthy',
        'potato-late_blight',
        'raspberry-healthy',
        'soybean-healthy',
        'squash-powdery_mildew',
        'strawberry-healthy',
        'strawberry-leaf_scorch',
        'tomato-bacterial_spot',
        'tomato-early_blight',
        'tomato-healthy',
        'tomato-late_blight',
        'tomato-leaf_mold',
        'tomato-septoria_leaf_spot',
        'tomato-spider_mites two-spotted_spider_mite',
        'tomato-target_spot',
        'tomato-tomato_mosaic_virus',
        'tomato-tomato_yellow_leaf_curl_virus'
    ) 

    NUM_CLASSES = len(CLASS_NAMES) 

    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all", 
            val_split: float = 0.2,
            test_split: float = 0.2,
            random_seed: int = 69,
            image_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
            **kwargs
            ) -> None:

        assert split in ("train", "val", "trainval", "test", "all"), "invalid split"
        _reproducibility_kwargs = {
            "random_seed": random_seed,
            "val_split": val_split,
            "test_split": test_split
        }

        self.root = root
        self.split = split
        self.df = df if isinstance(df, DataFrame) else self.equal_test_df(root, **_reproducibility_kwargs)
        self.df = self.df.assign(image_path = lambda df: df["image_path"].apply(lambda x: str(x)))
        self.split_df = self.df.assign(df_idx = lambda df: df.index).pipe(self.subset_df, split).pipe(self.prefix_root, self.root)
        self.image_transform = image_transform or self.DEFAULT_IMAGE_TRANSFORM
        self.common_transform = common_transform or self.DEFAULT_COMMON_TRANSFORM

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, idx) -> tuple[Tensor, int, int]:
        row_idx = self.split_df.iloc[idx]
        image = imread(row_idx["image_path"]).squeeze()
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = self.image_transform(image)
        if self.split == "train":
            image = self.common_transform(image)
        return image, row_idx["label_idx"], row_idx["df_idx"]

    @staticmethod
    def rename_class_name(filename: str) -> str:
        filename = filename.lower()
        splits = filename.split('__')
        plant_name = splits[0].replace(',', '').removesuffix('_')
        disease_name = splits[-1].removeprefix('_')
        return f"{plant_name}-{disease_name}"

    @classmethod    
    def stratified_df(cls, root: Path, val_split: float, test_split: float, random_seed: int) -> DataFrame: 
        df = DataFrame({"name": list(root.rglob("*.JPG")) + list(root.rglob("*.jpg")) + list(root.rglob("*.PNG"))})
        df["image_path"] = df["name"].apply(lambda x: Path(x.parent.stem, x.name)) # type: ignore
        df["label_str"] = df["name"].apply(lambda x: cls.rename_class_name(x.parent.stem))
        df["label_idx"] = df["label_str"].apply(lambda x: cls.CLASS_NAMES.index(x))

        test = (df
                .groupby("label_str", group_keys=False)
                .apply(lambda x: x.sample(frac=test_split, random_state=random_seed, axis=0))
                .assign(split = "test"))
        
        val = (df
                .drop(test.index, axis = 0)
                .groupby("label_str", group_keys=False)
                .apply(lambda x: x.sample(frac=val_split, random_state=random_seed, axis=0))
                .assign(split = "val"))

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        return (concat([train, val, test])
                .sort_values("image_path")
                .reset_index(drop = True)
                .drop("name", axis = 1))
    
    @classmethod
    def equal_test_df(cls, root: Path, val_split: float, test_split: float, random_seed: int) -> DataFrame: 
        df = DataFrame({"name": list(root.rglob("*.JPG")) + list(root.rglob("*.jpg")) + list(root.rglob("*.PNG"))})
        df["image_path"] = df["name"].apply(lambda x: Path(x.parent.stem, x.name)) # type: ignore
        df["label_str"] = df["name"].apply(lambda x: cls.rename_class_name(x.parent.stem))
        df["label_idx"] = df["label_str"].apply(lambda x: cls.CLASS_NAMES.index(x))

        test = (df
                .groupby("label_str", group_keys=False)
                .apply(lambda x: x.sample(n = 100, random_state=random_seed, axis=0))
                .assign(split = "test"))
        
        val = (df
                .drop(test.index, axis = 0)
                .groupby("label_str", group_keys=False)
                .apply(lambda x: x.sample(frac=val_split, random_state=random_seed, axis=0))
                .assign(split = "val"))

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        return (concat([train, val, test])
                .sort_values("image_path")
                .reset_index(drop = True)
                .drop("name", axis = 1))

    def subset_df(self, df: DataFrame, split: str) -> DataFrame:
        if split == "all":
            return df
        elif split == "trainval":
            return df[(df["split"] == "train") | (df["split"] == "val")].reset_index(drop=True)
        return df[df["split"] == split].reset_index(drop=True)

    def prefix_root(self, df: DataFrame, root: Path) -> DataFrame:
        return df.assign(image_path = lambda df: df["image_path"].apply(lambda x: str(root/x)))