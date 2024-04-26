from typing import Optional

import pytest
import tempfile
from pathlib import Path

class TestDataset:
    #imagefolder_dataset = None # type: ignore
    #hdf5_dataset = None # type: ignore
    #imagefolder_dataset_valid_root: Optional[str] = None
    #hdf5_dataset_valid_root: Optional[str] = None

    def test_imagefolder_invalid_root(self):
        # Test if :root is a nonexistant path
        _dataset = self.imagefolder_dataset
        with pytest.raises(OSError):
            _dataset(root = "/tmp/this_dir_should_not_exist/")

        # Test if :root is a file
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(OSError):
                _dataset(root = temp_file.name)

        # Test if :root is an empty directory 
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(OSError):
                _dataset(root = temp_dir)


#def imagefolder_valid_root():
    #dataset = imagefolder_dataset
    #valid_root = imagefolder_dataset_valid_root
    #assert dataset(root = valid_root).root == Path(valid_root)

#@pytest.mark.parametrize("invalid_split", ("", "training", "validation", "testing"))
#def imagefolder_invalid_splits(invalid_split):
    #dataset = imagefolder_dataset
    #valid_root = imagefolder_dataset_valid_root
    #with pytest.raises(ValueError):
        #dataset(root = valid_root, split = invalid_split)

#@pytest.fixture(params = ("train", "val", "test", "trainval", "all"))
# def dataset_valid_splits(dataset, valid_root):
    # for valid_split in ("train", "val", "test", "trainval", "all"):
        # assert dataset(root = valid_root, split = valid_split).split == valid_split