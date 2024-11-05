from typing import Optional
from pathlib import Path

class FileSystemIO:

    @classmethod
    def get_valid_file_err(cls, *args) -> Path:
        """returns resolved file path if the path pointed to by :args exists on the local fs and is a file."""
        file_path = cls.get_absolute_path(*args)
        if not file_path.is_file():
            raise FileNotFoundError(f"{file_path} does not point to a local file")
        return file_path

    @classmethod
    def is_valid_file(cls, *args: tuple) -> bool:
        try:
            cls.get_valid_file_err(*args)
            return True 
        except OSError:
            return False
    
    @classmethod
    def get_valid_dir_err(cls, *args, empty_ok: bool = False) -> Path:
        """returns resolved dir path if the path pointed to by :args exists on the local fs and is a dir. optionally checks
        if dir is non-empty, otherwise raises OSError"""
        dir_path = cls.get_absolute_path(*args)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_path} does not point to a local directory")
        if not empty_ok and not bool(list(dir_path.iterdir())):
            raise OSError(f"{dir_path} is an empty directory")
        return dir_path

    @classmethod 
    def is_valid_dir(cls, *args, empty_ok: bool = False) -> bool:
        try:
            cls.get_valid_dir_err(*args, empty_ok=empty_ok)
            return True
        except OSError:
            return False

    @classmethod
    def get_new_dir(cls, *args) -> Path:
        """creates a new directory and its parents if they don't exist, and returns it's path. No error is raised if dir exists"""
        dir_path = cls.get_absolute_path(*args)
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path
    
    @staticmethod
    def get_absolute_path(*args: tuple) -> Path:
        return Path(*args).resolve().expanduser()