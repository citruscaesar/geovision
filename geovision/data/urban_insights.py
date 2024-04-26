from abc import ABC, abstractmethod
from pandas import DataFrame
from .dataset import Dataset

class BuildingFootprintImagefolderSegmentation(Dataset):
    pass

class BuildingFootprintHDF5Segmentation(Dataset):
    pass

class BuildingFootprintDataset(ABC):
    @classmethod
    @abstractmethod
    def download_from_src(cls, local) -> DataFrame:
        # http or s3, asyncio or multiprocessing, zipfile extraction, etc
        pass

    def transform_to_imagefolder(cls, local) -> None:
        # downloaded files -> imagefolder 
        pass

    @classmethod
    @abstractmethod
    def transform_to_hdf(cls, local) -> None:
        # downloaded files -> hdf5
        pass

    @classmethod
    @abstractmethod
    def get_crs_data_df(cls, local) -> DataFrame:
        # returns crs data by loading crs_data.csv if found or regenerate if possible / required
        pass

    @classmethod
    @abstractmethod
    def get_dataset_df(cls) -> DataFrame:
        # returns dataset_df compliant with dataset_df schema for building segmentation
        pass
