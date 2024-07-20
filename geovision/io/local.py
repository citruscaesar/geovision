from typing import Optional
from pathlib import Path

def is_dir_path(*args) -> bool:
    return Path(*args).suffix == ''

def is_valid_dir(*args) -> bool:
    return Path(*args).is_dir()

def is_empty_dir(*args) -> bool:
    """returns True if target dir is empty, False if non-empty. Raises OSError if dir is invalid"""
    dir_path = Path(*args).resolve().expanduser()
    if not bool(list(dir_path.iterdir())):
        return True
    return False

def is_valid_file(*args) -> bool:
    try:
        get_valid_file_err(*args)
        return True 
    except OSError:
        return False

def is_hdf5_file(*args) -> bool:
    return Path(*args).suffix in (".h5", ".hdf5")

def is_archive_file(*args) -> bool:
    return Path(*args).suffix in (".zip", ".tgz", ".7z")

def get_new_dir(*args) -> Path:
    """creates a new directory and its parents if they don't exist, and returns it's path. No error is raised if dir exists"""
    dir_path = Path(*args).resolve().expanduser()
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path

def get_valid_dir_err(*args, empty_ok: bool = False) -> Path:
    """returns resolved dir path if dir exists on the local fs and optionally checks if dir is non-empty, otherwise raises OSError"""
    dir_path = Path(*args).resolve().expanduser()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} does not point to a local directory")
    if not empty_ok and is_empty_dir(dir_path):
        raise OSError(f"{dir_path} is an empty directory")
    return dir_path

def get_valid_file_err(*args) -> Path:
    """returns resolved file path if file exists on the local fs, otherwise raises OSError"""
    file_path = Path(*args).resolve().expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} does not point to a local file")
    return file_path

def get_experiments_dir(config) -> Path:
    """generates and returns path to experiments log dir, ~/experiments/{ds_name}_{ds_task}/{config.name}"""
    ds_name, _, ds_task = config.dataset_.name.split('_')
    expr_name = config.name.replace(' ', '_')
    return get_new_dir(Path.home(), "experiments", '_'.join([ds_name, ds_task]), expr_name)

# TODO: refactor for train.py
def get_ckpt_path(config, epoch: int = -1, step: int = -1) -> Optional[Path | str]:
    def display_ckpts_list(ckpts) -> None:
        names = [ckpt.name for ckpt in ckpts]
        names[-1] = f"*{names[-1]}"
        display(f"found ckpts: {names}") # noqa # type: ignore

    if epoch < -1:
        raise ValueError(f"epoch must be >= -1, got{epoch}")
    if step < -1:
        raise ValueError(f"step must be >= -1, got {step}")

    ckpts = sorted(get_experiments_dir(config).rglob("*.ckpt"))
    if len(ckpts) == 0:
        print("no ckpt found in experiments, returning None")
    elif epoch == -1 and step == -1:
        display_ckpts_list(ckpts)
        return ckpts[-1] 
    elif epoch != -1 and step == -1:
        ckpts = [ckpt for ckpt in ckpts if f"epoch={epoch}" in ckpt.name]
        display_ckpts_list(ckpts)
        return ckpts[-1]
    else:
        ckpts = [ckpt for ckpt in ckpts if f"epoch={epoch}_step={step}" in ckpt.name]
        if len(ckpts) == 0:
            print("found no matching ckpt, returning None")
            return None
        else:
            display_ckpts_list(ckpts)
            return ckpts[-1]