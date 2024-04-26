from .imagenette import (
    ImagenetteImagefolderClassification,
    ImagenetteHDF5Classification 
)

DATASETS = {
    "imagenette_imagefolder_classification": ImagenetteImagefolderClassification,
    "imagenette_hdf5_classification": ImagenetteHDF5Classification
}

def get_dataset_from_key(key: str):
    if key in DATASETS.keys():
        return DATASETS[key]
    else:
        raise NotImplementedError(f"{key} has not been implemented, ref. geovision.data.DATASETS")

def get_dataset_keys():
    return sorted(DATASETS.keys())