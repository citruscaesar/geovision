from typing import Optional, Iterable

import requests

from asyncio import as_completed
from aiofiles import open as aiopen
from aiohttp import ClientSession, ClientTimeout
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse

class S3IO:
    access_key = None
    secret_key = None
    endpoint_url = None
    no_sign_request = True

    def __init__(self, access_key: str, secret_key: str, endpoint_url: str):
        # use :keys, if not provided, look into environment_variables, otherwise use --no-sign-request
        pass

    def list_contents(self, remote: Path):
        # display tree with contents and file sizes in MB
        pass

    def download_item(remote: Path, local: Path):
        pass

    def upload_item(local: Path, remote: Path):
        pass

    def sync_dir(remote: Path, local: Path):
        pass

    def clear_failed_uploads(remote: Path):
        pass

class HTTPIO:
    @staticmethod
    def get_filename_from_url(remote: str) -> str:
        return urlparse(remote).path.split('/')[-1]

    @classmethod
    def download_url(cls, remote: str, local: Path):
        r"""Download a file from :remote to :local over HTTP. If :local points to a directory, it is
        created if it does not exist and the filename is inferred from :remote."""

        local = local.resolve().expanduser()
        Path.mkdir(local, exist_ok=True, parents=True)
        filename = cls.get_filename_from_url(remote)
        local = local / filename 

        response = requests.get(remote, stream = True)
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total = total_size, unit = "B", unit_scale = True, 
                  desc = f"Downloading {local.name}") as progress_bar:
            with open(local, 'wb') as file:
                print(f"Downloading {remote} to {local}")
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("could not download file completely")
    
    @classmethod
    async def async_download_urls(cls, remote: Iterable[str], local: Path) -> None:
        r"""Download files from :remote and save to :local asynchronously over HTTP. :local must 
        be a directory path, will be created if dosen't exist"""
        
        async def _async_download_url(remote: str, local: Path, session: ClientSession):
            local = local / cls.get_filename_from_url(remote)
            async with session.get(remote, ssl = False) as response:
                async with aiopen(local, 'wb') as file:
                    async for data in response.content.iter_chunked(1024):
                        await file.write(data)

        local = local.expanduser()
        if not is_dir_path(local):
            raise ValueError(f"{local} is not a directory path")
        local.mkdir(exist_ok=True, parents=True)

        async with ClientSession(timeout = ClientTimeout(total = None)) as session:
            downloads = [_async_download_url(url, local, session) for url in remote]
            for download in tqdm(
                as_completed(downloads), total = len(downloads), desc = "Downloading URLs"
            ):
                await download