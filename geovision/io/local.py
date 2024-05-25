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
    # TODO: change this, if file path does not match exactly, is_file will fail anyways
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} does not point to a local file")
    if valid_extns is not None:
        if file_path.suffix not in valid_extns:
            raise OSError(f"{file_path} must be one of {valid_extns}")
    return file_path

def get_experiments_dir(config) -> Path:
    ds_name, _, ds_task = config.dataset.name.split('_')
    expr_name = config.name.replace(' ', '_')
    return get_new_dir(Path.home(), "experiments", '_'.join([ds_name, ds_task]), expr_name)

def get_dataset_dir(root: str | Path, dir_name: str, invalid_ok = False) -> Path:
    def _validate(*args) -> Path:
        dir_path = Path(root).expanduser().resolve() / Path(*args) 
        if invalid_ok:
            return get_new_dir(dir_path)
        return get_valid_dir_err(dir_path)

    match dir_name:
        case "archives":
            return _validate("archives")
        case "imagefolder":
            return _validate("imagefolder")
        case "hdf5":
            return _validate("hdf5")
        case "images":
            return _validate("imagefolder", "images")
        case "masks":
            return _validate("imagefolder", "masks")
        case _:
            raise ValueError("invalid :dir_name, must be one of archives, imagefolder, hdf5, images, masks")

def get_ckpt_path(config, epoch: int = -1, step: int = -1) -> Optional[Path]:
    try:
        experiments_dir = get_valid_dir_err(get_experiments_dir(config) / "checkpoints", empty_ok=False)
        if epoch == -1 and step == -1:
            ckpts = sorted(experiments_dir.glob("epoch=*_step=*.ckpt"))
            display(ckpts) # noqa # type: ignore
            return ckpts[-1]
        elif epoch != -1 and step == -1:
            ckpts = sorted(experiments_dir.glob(f"epoch={epoch}_step=*.ckpt"))
            display(ckpts) # noqa # type: ignore
            return ckpts[-1]
        else:
            return get_valid_file_err(experiments_dir, f"epoch={epoch}_step={step}.ckpt")
    except OSError:
        display("no matching ckpt found, returning None") # noqa # type: ignore
        return None
