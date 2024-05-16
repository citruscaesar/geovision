from typing import Optional
from pathlib import Path

def is_dir_path(*args) -> bool:
    return Path(*args).suffix == ''

def is_valid_dir(*args) -> bool:
    return Path(*args).is_dir()

def is_empty_dir(*args) -> bool:
    return not bool(list(Path(*args).iterdir()))

def is_valid_file(*args, valid_extns: Optional[tuple[str, ...]] = None) -> bool:
    try:
        get_valid_file_err(*args, valid_extns=valid_extns)
        return True 
    except OSError:
        return False

def is_hdf5_file(*args) -> bool:
    return Path(*args).suffix in (".h5", ".hdf5")

def is_archive_file(*args) -> bool:
    return Path(*args).suffix in (".zip", ".tgz", ".7z")

def get_new_dir(*args) -> Path:
    dir_path = Path(*args).resolve().expanduser()
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path

def get_valid_dir_err(*args, empty_ok: bool = False) -> Path:
    dir_path = Path(*args).resolve().expanduser()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} does not point to a local directory")
    if not empty_ok:
        if not bool(list(dir_path.iterdir())):
            raise OSError(f"{dir_path} is an empty directory")
    return dir_path

def get_valid_file_err(*args, valid_extns: Optional[tuple[str, ...]] = None) -> Path:
    file_path = Path(*args).resolve().expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} does not point to a local file")
    if valid_extns is not None:
        if file_path.suffix not in valid_extns:
            raise OSError(f"{file_path} must be one of {valid_extns}")
    return file_path
