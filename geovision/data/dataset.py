from typing import Literal, Optional

from abc import ABC, abstractmethod 
from enum import StrEnum 
from pandas import DataFrame, concat
from pandera import DataFrameSchema
from torchvision.transforms.v2 import Transform # type: ignore
from pathlib import Path
from geovision.io.local import (
    is_valid_dir, 
    is_empty_dir, 
    is_valid_file,
    is_hdf5_file
)

SPLITS = ("train", "val", "test", "trainval", "all")

class Dataset(ABC):
    def _validate_and_set_root_dir(self, root) -> None:
        """sets self._root if it's a path to a non-empty local directory
        Parameters
        -
        :root -> argument to pathlib.Path, str or os.PathLike 

        Raises
        -
        OSError if not a path or not empty
        *other errors possible from pathlib
        """
        if not is_valid_dir(root):
            raise OSError(f":root is not a directory, got {root}")
        if is_empty_dir(root):
            raise OSError(":root is empty")
        self._root = Path(root)
    
    def _validate_and_set_root_hdf(self, root) -> None:
        """sets self._root if it's a local hdf5 file
        Parameters
        -
        :root -> argument to pathlib.Path, str or os.PathLike 

        Raises
        -
        OSError if not a path or not empty
        *other errors possible from pathlib
        """
        if is_valid_file(root):
            raise OSError(":root is not a file")
        if is_hdf5_file(root):
            raise OSError(":root is not an HDF5 file, suffix must be .h5 or .hdf5")
        self._root = Path(root)
    
    def _validate_and_set_split(self, split) -> None:
        """sets self._split if it's valid, otherwise raises ValueError"""
        if split not in SPLITS:
            raise ValueError(f"{split} is invalid for :split, must be one of {SPLITS}")
        self._split = split
    
    def _validate_and_get_transform(self, transform, default_transform) -> Transform:
        """returns :transform if it's a torchvision.Transform and default_transform if it's None
        Parameters
        -
        :transform -> torchvision.Transform or None

        Raises
        -
        TypeError
        """
        if isinstance(transform, Transform):
            return transform
        elif transform is None:
            return default_transform
        else:
            TypeError(f":transform is invalid, should be Transform or None, got {type(transform)}")
        
    def _validate_and_set_image_transform(self, image_transform, default) -> None:
        self._image_transform = self._validate_and_get_transform(image_transform, default) 

    def _validate_and_set_target_transform(self, target_transform, default) -> None:
        self._target_transform = self._validate_and_get_transform(target_transform, default)

    def _validate_and_set_common_transform(self, common_transform, default) -> None:
        self._common_transform = self._validate_and_get_transform(common_transform, default)       
        
    def _validate_and_set_df(self, df: DataFrame, schema: DataFrameSchema) -> None:
        self._df = schema(df, inplace = True) 

    def _validate_and_set_split_df(self, df: DataFrame, split: str, root: Path, schema: DataFrameSchema) -> None:
        self._split_df = schema(
            df
            .assign(df_idx = lambda df: df.index)
            .pipe(self._get_split_df, split)
            .pipe(self._get_root_prefixed_to_df_paths, root), 
        inplace = True) 

    def _validate_and_get_split_params(
            self, 
            test_split, 
            val_split, 
            random_seed
        ) -> tuple[float, float, int]:
        """return :split_params if valid otherwise raises ValueError or TypeError"""
        if not isinstance(test_split, int | float):
            raise TypeError(f":test_split must be numeric, got {type(test_split)}")
        if not isinstance(val_split, int | float):
            raise TypeError(f":test_split must be numeric, got {type(test_split)}")
        if not isinstance(random_seed, int):
            raise TypeError(f":random seed must be an integer, got {type(random_seed)}")
        if not (test_split < 1 and test_split >= 0):
            raise ValueError(f":test_split must belong to [0, 1), received {test_split}")
        if not (val_split < 1 and val_split >= 0):
            raise ValueError(f":val_split must belong to [0, 1), received {val_split}")
        if not (test_split + val_split < 1):
            raise ValueError(f":test_spilt = {test_split} + :val_split = {val_split} must be < 1")
        return test_split, val_split, random_seed
    
    def _get_random_split_df(self, df: DataFrame, random_seed: int):
        pass

    def _get_stratified_split_df(
            self,
            df: DataFrame, 
            split_col: str,
            split_params: tuple[float, float, int]
        ) -> DataFrame:
        """returns df split into train-val-test, by stratified(proportionate) sampling, based on
        :split_col and :split_params 
        s.t. num_eval_samples[i] = eval_split * num_samples[i], i -> {0, ..., num_classes-1}.

        Parameters
        - 
        :df -> dataframe to split
        :split_col -> column in df used to 
        :split_params -> (test_split, val_split, random_seed), used to sample random rows from df 
        
        Returns
        -
        DataFrame 

        Raises
        -
        AssertionError if :split_col is not present in :df
        """ 
        assert split_col in df.columns, f"invalid schema, df does not contain {split_col}"
        test_split, val_split, random_seed = split_params
        test = (df
                .groupby(split_col, group_keys=False)
                .apply(lambda x: x.sample(frac = test_split, random_state = random_seed, axis = 0), 
                        include_groups = False) # type: ignore
                .assign(split = "test"))

        val = (df
                .drop(test.index, axis = 0)
                .groupby(split_col, group_keys=False)
                .apply(lambda x: x.sample(frac = val_split / (1-test_split), 
                                          random_state = random_seed, axis = 0), 
                        include_groups = False) # type: ignore
                .assign(split = "val"))

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        return (concat([train, val, test])
                .sort_index()
                .drop(split_col, axis = 1))

    def _get_split_df(self, df: DataFrame, split: str) -> DataFrame:
        """returns a subset of rows where :split matches :df.split.

        Parameters
        - 
        :df -> dataframe to subset.\n
        :split -> parameter to match with the df["split"] column.\n

        Raises
        -
        AssertionError -> if :df does not contain a column named 'split'
        """

        assert "split" in df.columns, "invalid schema, df does not contain a column named 'split'"
        if split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == split].reset_index(drop=True))
    
    def _get_root_prefixed_to_df_paths(self, df: DataFrame, root: Path) -> DataFrame:
        r"""prefixes :root to the image_path and mask_path columns in :df

        Parameters 
        -
        :df -> dataframe 
        :root -> path prepended to df paths

        Raises
        -
        No Error -> if either column is missing from the dataframe
        """
        if "image_path" in df.columns:
            df["image_path"] = df["image_path"].apply(lambda x: str(Path(root, x)))
        if "mask_path" in df.columns:
            df["mask_path"] = df["mask_path"].apply(lambda x: str(Path(root, x)))
        return df

    def __repr__(self) -> str:
        return '\n'.join([
            f"{self.name} dataset for {self.task}",
            f"local @ [{self.root}]",
            f"with {len(self.class_names)} classes: {self.class_names}",
            f"and {len(self)} images loaded under {self.split} split",
        ])

    @abstractmethod
    def __init__(
            self,
            root: Path,
            df: Optional[DataFrame] = None,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            test_split: float = 0.20,
            val_split: float = 0.10,
            random_seed: int = 42,
            image_transform: Optional[Transform] = None,
            target_transform: Optional[Transform] = None,
            common_transform: Optional[Transform] = None,
    ) -> None:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def root(self) -> Path:
        pass

    @property
    @abstractmethod
    def split(self) -> Literal["train", "val", "trainval", "test", "all"]:
        pass

    @property
    @abstractmethod
    def df(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def split_df(self) -> DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def task(self) -> str:
        pass

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def means(self) -> tuple[float, ...]:
        pass

    @property
    @abstractmethod
    def std_devs(self) -> tuple[float, ...]:
        pass
