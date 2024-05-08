import pytest
import tempfile

def test_config(dataset, root):
    print(f"""\nTESTING: {dataset}\nROOT: {root}""")

def test_default_init(dataset, root):
    ds = dataset(root)
    assert ds.root == root 
    assert ds.split == "all"

@pytest.mark.parametrize("valid_split", ("train", "val", "test", "trainval", "all"))
def test_valid_split(dataset, root, valid_split):
    assert dataset(root = root, split = valid_split).split == valid_split

def test_valid_transforms(dataset, root):
    pass

def test_invalid_root_error(dataset, storagetype):
    # Test if :root is a nonexistant path
    with pytest.raises(OSError):
        dataset(root = "/tmp/this_dir_should_not_exist/")

    # Test if :root is an empty directory 
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(OSError):
            dataset(root = temp_dir)
        
    # Test if :root is not .h5 or .hdf5 file
    with tempfile.NamedTemporaryFile() as temp_file:
        with pytest.raises(OSError):
            dataset(root = temp_file.name)

    #TODO: add tests for specific storage types

@pytest.mark.parametrize("invalid_split", ("", "training", "validation", "testing"))
def test_invalid_split_error(dataset, root, invalid_split):
    with pytest.raises(ValueError):
        dataset(root = root, split = invalid_split)

def test_invalid_split_params_error(dataset, root):
    # Invalid :split_params raise TypeError or ValueError
    pass

def test_invalid_df_error(dataset, root):
    # Invalid :df raises SchemaError
    pass

def test_invalid_transforms_error(dataset, root):
    pass
