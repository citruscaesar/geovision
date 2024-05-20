import shutil
from pathlib import Path

import requests

import aiohttp
import aiofiles
import asyncio

import zipfile
import py7zr
import multivolumefile

import numpy as np
import pandas as pd

from tqdm.asyncio import tqdm, tqdm_asyncio

DATA = Path("/media/sambhav/30AC4696AC46568E/datasets/urban-feature-extraction")

class DatasetDownloader:
    def __init__(self, root:Path, urls:dict):
        if not (root.exists() and root.is_dir()):
            root.mkdir(parents=True)
        self.root = root 

        # src_urls is a dict such that
        # src_urls[filename:str]  = url:str
        self.src_urls = urls

        # init required directories
        self.download_dir = self.root / "downloads"
        self.download_dir.mkdir(exist_ok=True)

    async def download_one_file(self, session, url:str, file_path:Path):
        """Download one file from url and save to disk at file_path"""
        async with session.get(url, ssl = False) as r:
            total_size = int(r.headers.get('content-length', 0))
            async with aiofiles.open(file_path, "wb") as f:
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
                async for chunk in r.content.iter_any():
                    await f.write(chunk)
                    progress_bar.update(len(chunk))

    async def download_files(self) -> None:
        #Download files from self.src_urls, skip if already_downloaded
        timeout = aiohttp.ClientTimeout(total = None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            coroutines = list()
            for file_name, url in self.src_urls.items():
                file_path = self.download_dir / file_name 
                coroutines.append(self.download_one_file(session, url, file_path))
            await asyncio.gather(*coroutines)   

class InriaETL:
    urls = ["https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004",
            "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005"]
    
    def __init__(self, root: Path):
        self.root = root

    def Download(self):
        downloader = DatasetDownloader(root = self.root,
                                       urls = {Path(url).name : url for url in self.urls})
        asyncio.run(downloader.download_files())


inria = InriaETL(DATA / "inria")
inria.Download()