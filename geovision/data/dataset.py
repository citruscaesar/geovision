from typing import Literal, Optional, Any

from pathlib import Path
from pandas import DataFrame, read_csv, concat
from pandera import DataFrameSchema, Column, Check
from abc import ABC, abstractmethod 
from pydantic import BaseModel, ConfigDict
from torchvision.transforms.v2 import Transform # type: ignore
from geovision.io.local import get_valid_dir_err, get_valid_file_err
from geovision.logging import get_logger
from tqdm import tqdm

logger = get_logger("dataset")

class DatasetConfig(BaseModel):
    random_seed: int | None = None
    test_sample: int | float | None = None
    val_sample: int | float | None = None
    tabular_sampling: str | None = None

    tile_x: tuple | None = None
    tile_y: tuple | None  = None
    spatial_sampling: str | None = None

    bands: tuple | None = None
    spectral_sampling: str | None = None

    temporal_sampling: str | None = None

class TransformsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_transform: Transform | None = None
    target_transform: Transform | None = None
    common_transform: Transform | None = None

class Dataset(ABC):
    valid_splits = ("train", "val", "test", "trainval", "all")

    def __repr__(self) -> str:
        name, storage, task = self.name.split('_')
        return '\n'.join([
            f"{name} dataset for {task}",
            f"local {storage} @ [{self.root}]",
            f"with {len(self.class_names)} classes: {self.class_names}",
            f"and {len(self)} images loaded under {self.split} split",
        ])

    @abstractmethod
    def __init__(
            self,
            root: Path,
            split: Literal["train", "val", "trainval", "test", "all"] = "all",
            df: Optional[DataFrame] = None,
            config: Optional[DatasetConfig] = None,
            transforms: Optional[TransformsConfig] = None,
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
    def transforms(self) -> TransformsConfig:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def means(self) -> tuple[float, ...]:
        pass

    @property
    @abstractmethod
    def std_devs(self) -> tuple[float, ...]:
        pass

class Validator:
    @staticmethod
    def _get_unit_interval(num, name: str = "") -> float:
        if num is None:
            raise TypeError(f"{name} value cannot be None")
        if not isinstance(num, float):
            raise TypeError(f"{name} value expected to be Float, got {type(num)}")
        if not (num >= 0 and num < 1):
            raise ValueError(f"{name} value expected to belong to [0, 1), received {num}")
        return float(num)
        
    @staticmethod
    def _get_float(num, name: str = "") -> float:
        if num is None:
            raise TypeError(f"{name} value cannot be None")
        if not isinstance(num, float | int):
            raise TypeError(f"{name} value expected to be Numeric, got {type(num)}")
        return float(num) 
    
    @staticmethod
    def _get_int(num, name: str = "") -> int:
        if num is None:
            raise TypeError(f"{name} value cannot be None")
        if not isinstance(num, float | int):
            raise TypeError(f"{name} value expected to be Numeric, got {type(num)}")
        return int(num)

    @staticmethod
    def _get_root_dir(root: str | Path) -> Path:
        """returns :root if it's a path to an existing non-empty local directory, otherwise raises 
        OSError"""
        logger.info(f":root = {root}")
        return get_valid_dir_err(root, empty_ok=False) 
    
    @staticmethod
    def _get_root_hdf5(root: str | Path) -> Path:
        """returns :root if it's a local hdf5 file, otherwise raises OSError"""
        logger.info(f":root = {root}")
        return get_valid_file_err(root, valid_extns=(".h5", ".hdf5"))
    
    @staticmethod
    def _get_split(split: str) -> Literal["train", "val", "test", "trainval", "all"]:
        """returns :split if it's valid, otherwise raises ValueError"""
        logger.info(f":split = {split}")
        if split not in Dataset.valid_splits:
            raise ValueError(f":split must be one of {Dataset.valid_splits}, got {split}")
        return split # type: ignore
    
    @staticmethod
    def _get_transforms(
            transforms: Optional[TransformsConfig],
            default_transforms: TransformsConfig
        ) -> TransformsConfig:
        """returns a TransformsConfig with missing :transforms replaced with :default_transforms"""
        if transforms is None:
            logger.info("did not receive :transforms, using default transforms")
            transforms = default_transforms
        else:
            transforms = TransformsConfig(
                image_transform = transforms.image_transform or default_transforms.image_transform,
                target_transform = transforms.target_transform or default_transforms.target_transform,
                common_transform = transforms.common_transform or default_transforms.common_transform
            )
        logger.info(f"applying image_transforms: {transforms.image_transform}")
        logger.info(f"applying target_transforms: {transforms.target_transform}")
        logger.info(f"applying common_transforms: {transforms.common_transform}")
        return transforms
    
    @staticmethod
    def _get_df(
            df: Optional[str | Path | DataFrame],
            config: Optional[DatasetConfig],
            schema: DataFrameSchema,
            default_df: DataFrame,
            default_config: DatasetConfig
        ) -> DataFrame:
        logger.info("validating :df")
        if isinstance(df, DataFrame):
            logger.info(f"received :df {type(df)}")
            _df = df
        elif isinstance(df, str | Path):
            logger.info(f"received :df {type(df)}")
            _df = read_csv(get_valid_file_err(df, valid_extns=(".csv",)))
        elif df is None:
            logger.info("did not receive :df, using default df")
            if config is None:
                logger.info("did not receive :config, using default")
                _config = default_config
            else:
                logger.info(f"received :config {type(config)}")
                _config = config
            _df = (
                default_df
                .pipe(TabularSampler.sample, _config)
                .pipe(SpatialSampler.sample, _config)
                .pipe(SpectralSampler.sample, _config)
                .pipe(TemporalSampler.sample, _config)
            )
        else:
            raise TypeError(f":df must be one of pd.DataFrame, Path or None, got {type(df)}")
        #return _df
        return schema(_df, inplace = True) 
            
    @classmethod
    def _get_imagefolder_split_df(
            cls,
            df: DataFrame,
            schema: DataFrameSchema,
            root: Path,
            split: str,
        ) -> DataFrame:
        logger.info("validating imagefolder :split_df")
        _split_df = (
            df
            .assign(df_idx = lambda df: df.index)
            .pipe(cls._get_split_df, split)
            .pipe(cls._get_root_prefixed_to_df_paths, root)
        )
        return schema(_split_df, inplace=True)

    @classmethod
    def _get_hdf5_split_df(
            cls,
            df: DataFrame,
            schema: DataFrameSchema,
            split: str,
        ) -> None:
        _split_df = (
            df 
            .assign(df_idx = lambda df: df.index)
            .pipe(cls._get_split_df, split)
        )
        return schema(_split_df, inplace=True)

    @staticmethod
    def _get_split_df(df: DataFrame, split: str) -> DataFrame:
        """returns a subset of rows where :split matches :df.split.

        Parameters
        - 
        :df -> dataframe to subset.\n
        :split -> parameter to match with the df["split"] column.\n

        Raises
        -
        AssertionError -> if :df does not contain a column named 'split'
        """
        df = DataFrameSchema({"split": Column(str, Check.isin(Dataset.valid_splits))})(df)
        if split == "all":
            return df
        elif split == "trainval":
            return (df[(df.split == "train") | (df.split == "val")].reset_index(drop=True)) # type: ignore
        return (df[df.split == split].reset_index(drop=True))
    
    @staticmethod
    def _get_root_prefixed_to_df_paths(df: DataFrame, root: Path) -> DataFrame:
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

class TabularSampler:
    @classmethod
    def sample(cls, df: DataFrame, config: DatasetConfig) -> DataFrame:
        match config.tabular_sampling:
            case None:
                print(":tabular_sampling not specfied")
                return df
            case "stratified":
                return cls._get_stratified_split_df(
                    df, config.random_seed, config.test_sample, config.val_sample # type: ignore
                ) 
            case "equal":
                return cls._get_equal_split_df(
                    df, config.random_seed, config.test_sample, config.val_sample # type: ignore
                )
            case "random":
                return cls._get_random_split_df(df, config.random_seed) # type: ignore
            case _:
                raise ValueError(f"invalid :tabluar_sampling, must be one of stratified, equal, random or None, got {config.tabular_sampling}")

    @staticmethod
    def _get_stratified_split_df(
            df: DataFrame, 
            random_seed: int,
            test_sample: float,
            val_sample: float
        ) -> DataFrame:

        """returns df split into train-val-test, by stratified(proportionate) sampling, based on
        :split_col and :split_config 
        s.t. num_eval_samples[i] = eval_split * num_samples[i], i -> {0, ..., num_classes-1}.

        Parameters
        - 
        :df -> dataframe to split, must contain string typed column named "split_on"
        :random_seed -> to use for deterministic sampling from dataframe
        :test_sample -> proportion of samples per class for testing 
        :val_sample -> proportion of samples per class for validation
        
        Returns
        -
        DataFrame 

        Raises
        -
        SchemaError
        """ 
        df = DataFrameSchema({"split_on": Column(str)})(df)
        random_seed = Validator._get_int(random_seed, "random_seed")
        test_frac = Validator._get_unit_interval(test_sample, "test_sample") 
        val_frac = Validator._get_unit_interval(val_sample, "val_sample") 
        if not (test_frac + val_frac < 1):
            raise ValueError(f":test_sample + :val_sample = must be < 1, got {test_frac+val_frac}")

        test = (df
                .groupby("split_on", group_keys=False)
                .apply(lambda x: x.sample( # type: ignore
                        frac = test_frac, random_state = random_seed, axis = 0)
                    )
                .assign(split = "test")) 
               
        val = (df
                .drop(test.index, axis = 0)
                .groupby("split_on", group_keys=False)
                .apply(lambda x: x.sample( # type: ignore
                        frac = val_frac / (1-test_frac), random_state = random_seed, axis = 0)
                    )
                .assign(split = "val"))

        train = (df
                .drop(test.index, axis = 0)
                .drop(val.index, axis = 0)
                .assign(split = "train"))

        return (concat([train, val, test])
                .reset_index(drop = True))

    @staticmethod
    def _get_random_split_df(
            df: DataFrame, 
            random_seed: int
        ) -> DataFrame:
        return DataFrame()
    
    @staticmethod
    def _get_equal_split_df(
            df: DataFrame,
            random_seed: int,
            test_sample: int,
            val_sample: int,
        ) -> DataFrame:
        return DataFrame()

class SpatialSampler:
    @classmethod
    def sample(cls, df: DataFrame, config: DatasetConfig):
        match config.spatial_sampling:
            case None:
                return df
            case "image":
                pass
            case "geographic":
                pass

class SpectralSampler:
    @classmethod
    def sample(cls, df: DataFrame, config: DatasetConfig):
        match config.spectral_sampling:
            case None: 
                return df
            case "sen2rgb":
                pass
            case "sen2ms":
                pass

class TemporalSampler:
    @classmethod
    def sample(cls, df: DataFrame, config: DatasetConfig):
        match config.temporal_sampling:
            case None: 
                return df
            case "summer":
                pass