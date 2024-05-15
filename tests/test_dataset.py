import pytest

from pathlib import Path
from pandas import DataFrame
from torch import float32, uint8 
from numpy import integer
from torchvision.tv_tensors import Image, Mask
from geovision.io.local import get_new_dir 

def test_config(dataset_constructor, root):
    print(f"""\nTESTING: {dataset_constructor}\nROOT: {root}""")

def test_interface(dataset):
    # :root must be a path to a valid location in the FS
    assert isinstance(dataset.root, Path) and dataset.root.exists()
    # :split must be a str, initialized by default to "all"
    assert isinstance(dataset.split, str) and dataset.split == "all"
    # :df must ba a DataFrame, validated against dataframe schema 
    assert isinstance(dataset.df, DataFrame)
    # :split_df must be a DataFrame, define the number of samples in the current split and 
    # validated against dataframe split schema
    assert isinstance(dataset.split_df, DataFrame) and len(dataset) == len(dataset.split_df)
    # :name must be a string of the format datasetname_storagetype_task
    assert isinstance(dataset.name, str) and len(dataset.name.split('_')) == 3
    assert isinstance(dataset.class_names, tuple[str, ...])
    assert isinstance(dataset.num_classes, int)
    assert isinstance(dataset.means, tuple[float, ...])
    assert isinstance(dataset.std_devs, tuple[float, ...])

def test_invalid_root_error(dataset_constructor, storage, tmp_path):
    # Test if :root is a nonexistant path
    with pytest.raises(OSError):
        dataset_constructor(root = tmp_path / "tmp_nonexistant_root")

    # Test if :root is an empty directory 
    with pytest.raises(OSError):
        dataset_constructor(root = get_new_dir(tmp_path / "tmp_empty_root"))
        
    # Test if :root is not .h5 or .hdf5 file
    tmp_file = tmp_path / "tmp_dataset_file.txt"
    tmp_file.write_text("Temporary file for testing dataset root validation")
    with pytest.raises(OSError):
        dataset_constructor(root = tmp_file)
    
    # match storage:
        # case "imagefolder":
        # case "hdf5":

@pytest.mark.parametrize("invalid_split", ("", "training", "validation", "testing"))
def test_invalid_split_error(dataset_constructor, root, invalid_split):
    with pytest.raises(ValueError):
        dataset_constructor(root = root, split = invalid_split)

def test_default_split_is_all(dataset):
    assert dataset.split == "all"

@pytest.mark.parametrize("valid_split", ("train", "val", "test", "trainval", "all"))
def test_valid_split(dataset_constructor, root, valid_split):
    assert dataset_constructor(root = root, split = valid_split).split == valid_split

#def test_invalid_df_schema_error(dataset_constructor, root):
    #pass

def test_output(dataset, workflow):
    def _test_classification_output(out: tuple):
        assert isinstance(out[0], Image) 
        assert out[0].ndim == 3
        assert out[0].dtype == float32
        assert isinstance(out[1], integer)
        assert isinstance(out[2], integer)
    
    def _test_segmentation_output(out: tuple):
        assert isinstance(out, tuple) and len(out) == 3
        assert isinstance(out[0], Image) and out[0].ndim == 3
        assert out[0].ndim == 3
        assert out[0].dtype == float32
        assert isinstance(out[1], Mask) and out[1].ndim == 3
        assert out[0].ndim == 3
        assert out[0].dtype == uint8 
        assert isinstance(out[2], integer)
    
    match workflow:
        case "classification":
            _test_output = _test_classification_output
        case "segmentation":
            _test_output = _test_segmentation_output
        case _:
            raise ValueError

    for output in dataset:
        _test_output(output)