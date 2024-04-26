import pytest
from pathlib import Path
from geovision.data import get_dataset_from_key

def pytest_addoption(parser):
    parser.addoption(
        "--dataset", 
        action = "store_true", 
        default = "imagenette_imagefolder_classification",
        help = "Dataset Key, format = 'dataset_name_storagetype_task', e.g.imagenette_imagefolder_classification",
    )
    parser.addoption(
        "--root", 
        action = "store_true",
        dest = "root",
        default = "~/datasets/imagenette",
        help = "Path To Root Dir, e.g. ~/datasets/imagenette/"
    )

@pytest.fixture
def dataset(request):
    return get_dataset_from_key(request.config.getoption("--dataset"))

@pytest.fixture
def root(request):
    return Path(request.config.getoption("--root")).expanduser()

@pytest.fixture
def storagetype(request):
    dataset_key = request.config.getoption("--dataset")
    if "imagefolder" in dataset_key:
        return "imagefolder"
    elif "hdf5" in dataset_key:
        return "hdf5"
    elif "litdata" in dataset_key:
        return "litdata"