from typing import Literal, Optional

from abc import ABC, abstractmethod 
from pandas import DataFrame, concat, read_csv
from pandera import DataFrameSchema
from torchvision.transforms.v2 import Transform # type: ignore
from pathlib import Path
from geovision.config import DataFrameConfig, TransformConfig
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
        """
        _root = Path(root).expanduser().resolve()
        if not is_valid_dir(_root):
            raise OSError(f":root is not a directory, got {_root}")
        if is_empty_dir(_root):
            raise OSError(":root is empty")
        self._root = _root
    
    def _validate_and_set_root_hdf5(self, root) -> None:
        """sets self._root if it's a local hdf5 file
        Parameters
        -
        :root -> argument to pathlib.Path, str or os.PathLike 

        Raises
        -
        OSError if not a path or not an .h5 or .hdf5 file 
        *other errors possible from pathlib
        """
        if not is_valid_file(root):
            raise OSError(f":root is not a file, got {root}")
        if not is_hdf5_file(root):
            raise OSError(f":root is not an HDF5 file, suffix must be .h5 or .hdf5, got {root.suffix}")
        self._root = Path(root).expanduser()
    
    def _validate_and_set_split(self, split) -> None:
        """sets self._split if it's valid, otherwise raises ValueError"""
        if split not in SPLITS:
            raise ValueError(f"{split} is invalid for :split, must be one of {SPLITS}")
        self._split = split
    
    def _validate_and_set_transform(self, transform: Optional[TransformConfig], default: TransformConfig) -> None:
        if transform is None:
            self._image_transform = default.image_transform
            self._target_transform = default.target_transform
            self._common_transform = default.common_transform
        else:
            self._image_transform = transform.image_transform or default.image_transform
            self._target_transform = transform.target_transform or default.target_transform
            self._common_transform = transform.common_transform or default.common_transform

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
        
    def _validate_and_set_df(
            self,
            schema: DataFrameSchema,
            default_df = DataFrame,
            default_config = DataFrameConfig,
            split_col = str,
            df: Optional[DataFrame | Path] = None,
            config: Optional[DataFrameConfig] = None,
            ) -> None:
        if isinstance(df, DataFrame):
            _df = df
        elif isinstance(df, Path):
            _df = read_csv(df)
        elif df is None: 
            _config = default_config if config is None else config 
            _df = default_df 
            _df = DataFrameSplitter(_config).split(_df, split_col)
            if _config.tiling_strategy is not None:
                _df = DataFrameTiler(_config).tile(_df)
        else: 
            raise TypeError(f":df must be one of pd.DataFrame, Path or None, got {type(df)}")
        self._df = schema(_df, inplace = True) 

    def _validate_and_set_imagefolder_split_df(self, split: str, root: Path, schema: DataFrameSchema) -> None:
        self._split_df = schema(
            self._df 
            .assign(df_idx = lambda df: df.index)
            .pipe(self._get_split_df, split)
            .pipe(self._get_root_prefixed_to_df_paths, root), 
        inplace = True) 

    def _validate_and_set_hdf5_split_df(self, split: str, schema: DataFrameSchema) -> None:
        self._split_df = schema(
            self._df
            .assign(df_idx = lambda df: df.index)
            .pipe(self._get_split_df, split),
        inplace = True)

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
        if split == "all":
            return df
        elif split == "trainval":
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
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df: Optional[DataFrame] = None,
            df_config: Optional[DataFrameConfig] = None,
            transform: Optional[TransformConfig] = None,
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

class DataFrameSplitter:

    #TODO: find a more elegant way to do this, this is wayy too ugly
    #NOTE Interface: DataFrameSplitter(config).split(df, split_col)

    def __init__(self, config: DataFrameConfig):
        self.strategy = config.splitting_strategy
        match self.strategy:
            case None:
                raise TypeError(":splitting_strategy cannot be None")
            case "stratified":
                self._validate_and_set_stratified_split_config(config)
            case "random":
                self._validate_and_set_random_split_config(config)
            case _:
                raise ValueError(":splitting strategy must be one of stratified, random")

    def _validate_and_set_stratified_split_config(self, config: DataFrameConfig):
        if config.test_frac is None:
            raise TypeError("config.test_frac cannot be None")
        if config.val_frac is None:
            raise TypeError("config.val_frac cannot be None")
        if config.random_seed is None:
            raise TypeError("config.random_seed cannot be None")
        self.config = config
    
    def split(self, df: DataFrame, split_col: str) -> DataFrame:
        match self.strategy:
            case "stratified":
                return self._get_stratified_split_df(df, split_col)
            case "random":
                return self._get_random_split_df(df, split_col)
            case _:
                raise ValueError(f":splitting strategy must be one of stratified, random, got {self.strategy}")
    
    def _get_stratified_split_df(
            self,
            df: DataFrame, 
            split_col: str,
        ) -> DataFrame:

        """returns df split into train-val-test, by stratified(proportionate) sampling, based on
        :split_col and :split_config 
        s.t. num_eval_samples[i] = eval_split * num_samples[i], i -> {0, ..., num_classes-1}.

        Parameters
        - 
        :df -> dataframe to split
        :split_col -> column in df used to 
        
        Returns
        -
        DataFrame 

        Raises
        -
        AssertionError if :split_col is not present in :df
        """ 
        assert split_col in df.columns, f"invalid schema, df does not contain {split_col}"
        test_frac, val_frac, random_seed = self.config.test_frac, self.config.val_frac, self.config.random_seed 
        
        test = (df
                .groupby(split_col, group_keys=False)
                .apply(lambda x: x.sample(frac = test_frac, random_state = random_seed, axis = 0)) # type: ignore
                .assign(split = "test")) 
               
        val = (df
                .drop(test.index, axis = 0)
                .groupby(split_col, group_keys=False)
                .apply(lambda x: x.sample(frac = val_frac / (1-test_frac), # type: ignore
                                          random_state = random_seed, axis = 0))
                .assign(split = "val"))

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        return (concat([train, val, test])
                .reset_index(drop = True))

    def _validate_and_set_random_split_config(self, config: DataFrameConfig):
        if config.random_seed is None:
            raise TypeError("config.random_seed cannot be None")
        self.config = config

    def _get_random_split_df(
            self,
            df: DataFrame, 
            split_col: str,
        ) -> DataFrame:
        return DataFrame()
    
class DataFrameTiler:
    def __init__(self, config: DataFrameConfig):
        pass

    def tile(self, df: DataFrame) -> DataFrame:
        return DataFrame()