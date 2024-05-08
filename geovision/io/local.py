from pathlib import Path

def is_dir_path(*args) -> bool:
    return Path(*args).suffix == ''

def is_valid_dir(*args) -> bool:
    return Path(*args).is_dir()

def is_empty_dir(*args) -> bool:
    return not bool(list(Path(*args).iterdir()))

def is_valid_file(*args) -> bool:
    return Path(*args).is_file()

def is_hdf5_file(*args) -> bool:
    return Path(*args).suffix in (".h5", ".hdf5")

def is_archive_file(*args) -> bool:
    return Path(*args).suffix in (".zip", ".tgz", ".7z")

def get_valid_dir(*args) -> Path:
    dir_path = Path(*args)
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path