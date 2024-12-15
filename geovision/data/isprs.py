def transformation_strategy_vaihingen(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:
        
        images_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if "top/top_mosaic_09cm" in x and not x.endswith('/')])
            for image_src_path in tqdm(filenames, desc = "Vaihingen Image Files Progress"): 
                image_dst_path = image_dir/f"vaihingen{image_src_path.removeprefix('top/top_mosaic_09cm_area')}"
                extract_image(image_src_path, image_dst_path, inner)

        masks_zip_bytes = outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip")
        with zipfile.ZipFile(masks_zip_bytes) as inner:
            for filename in tqdm(sorted(inner.namelist()), desc = "Vaihingen Mask Files Progress"):
                mask_dst_path = mask_dir/f"vaihingen{filename.removeprefix('top_mosaic_09cm_area')}"
                # mask[:, :, 0] = vegetation and building 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = vegetation 
                mask = iio.imread(inner.open(filename, 'r')).squeeze() #type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dst_path, mask, extension=".tif")

def transformation_strategy_potsdam(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as outer:

        images_zip_bytes = outer.open("Potsdam/2_Ortho_RGB.zip")
        with zipfile.ZipFile(images_zip_bytes) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for image_src_path in tqdm(filenames, desc = "Potsdam Image Files Progress"):
                image_dest_filename = image_src_path.removeprefix("2_Ortho_RGB/top_potsdam_").removesuffix("_RGB.tif")
                image_dest_filename = image_dir/f"potsdam{''.join(image_dest_filename.split('_'))}.tif"
                extract_image(image_src_path, image_dest_filename, inner)
        
        masks_zip_path = outer.open("Potsdam/5_Labels_all.zip")
        with zipfile.ZipFile(masks_zip_path) as inner:
            filenames = sorted([x for x in inner.namelist() if x.endswith(".tif")])
            for mask_src_path in tqdm(filenames, desc = "Potsdam Mask Files Progress"): 
                mask_dest_filename = mask_src_path.removeprefix("top_potsdam_").removesuffix("label.tif")
                mask_dest_filename = mask_dir/f"potsdam{''.join(mask_dest_filename.split('_'))}.tif"

                # mask[:, :, 0] = background 
                # mask[:, :, 1] = building
                # mask[:, :, 2] = no idea
                mask = iio.imread(inner.open(mask_src_path)).squeeze() # type: ignore
                mask = mask[:, :, 1]
                mask = np.where(mask==255, 0, 255).astype(np.uint8)
                iio.imwrite(mask_dest_filename, mask, extension=".tif")



class ISPRSETL():
    urls = {
        "potsdam.zip": "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/",
        #"toronto.zip": "https://seafile.projekt.uni-hannover.de/f/fc62f9c20a8c4a34aea1/",
        "vaihingen.zip": "https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/",
    }
    password = "CjwcipT4-P8g"
    cookie_name = "sfcsrftoken"

    def __init__(self, root: Path):
        self.root_dir = root
        self.download_dir = root / "downloads"

        self.d_image_dir = self.download_dir / "images"
        self.d_mask_dir = self.download_dir / "masks"

        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"

        self.t_image_dir= self.root_dir / "test" / "images"
        self.t_mask_dir= self.root_dir / "test" / "masks"
    
    def download(self):
        downloader = DatasetDownloader(self.download_dir, self.urls)
        asyncio.run(downloader.download_files(self._download_file))

    def extract(self, low_storage_mode):
        extractor = DatasetExtractor 
        self._extract_vaihingen(extractor, low_storage_mode)
        self._extract_potsdam(extractor, low_storage_mode)
    
    def calculate_train_test_split(self):
        self.files_list = [x.name for x in self.d_image_dir.rglob("*.tif")]
        df = pd.DataFrame({"name": self.files_list})
        df["region"] = df.name.apply(lambda x:self._get_file_region(x))

        self.test_files_list = list()
        for region in df.region.unique():
            region_df = df[df["region"] == region]
            self.test_files_list += sorted(list(region_df.name))[:int(0.15*len(region_df))]
        
        self.train_files_list = sorted(list(set(self.files_list).difference(self.test_files_list)))
    
    def move_test_split(self):
        self.t_image_dir.mkdir(exist_ok=True, parents=True)
        self.t_mask_dir.mkdir(exist_ok=True, parents=True)

        for file_name in self.test_files_list:
            shutil.move((self.d_image_dir / file_name), self.t_image_dir)
            shutil.move((self.d_mask_dir / file_name), self.t_mask_dir)

    @staticmethod
    def read_mask(path:Path):
        mask = skimage.io.imread(path) # type: ignore
        mask = np.where(mask[:, :, 1] == 0, np.uint8(255), np.uint8(0))
        return np.expand_dims(mask, -1)

    def crop(self, window: int):
        self.image_dir.mkdir(exist_ok=True)
        self.mask_dir.mkdir(exist_ok=True)

        cropper = CropUtil()

        for file_name in self.train_files_list:

            cropped_image_view = cropper._crop_one_scene(
                tile_path = self.d_image_dir / file_name,
                window = window,
                read_scene = cropper._read_image
            )

            cropped_mask_view = cropper._crop_one_scene(
                tile_path = self.d_mask_dir / file_name,
                window = window,
                read_scene = self.read_mask
            )

            for image_crop, mask_crop in zip(cropped_image_view, cropped_mask_view):
                crop_name = str(uuid.uuid4())
                cropper._save_as_jpeg_100(image_crop, (self.image_dir / crop_name))
                cropper._save_as_jpeg_100(mask_crop.squeeze(), (self.mask_dir / crop_name))

    async def _download_file(self, session, url:str, file_path:str) -> None:
        async with session.get(url) as response:
                cookies = session.cookie_jar.filter_cookies(url)
                cookie_value = cookies[self.cookie_name].value

        #async with session.get(url, ssl = False) as get_request:
            #cookies = {self.cookie_name: get_request.cookies.get(self.cookie_name)}
        payload = {"csrfmiddlewaretoken": cookie_value, 
                   "password": self.password}

        async with session.post(url+"?dl=1", data = payload) as r:
           async with aiofiles.open(file_path, "wb") as f:
                async for chunk in r.content.iter_any():
                    await f.write(chunk)

    def _extract_vaihingen(self, extractor, low_storage_mode: bool):
        vaihingen_zip_path = self.download_dir / "vaihingen.zip"
        image_dataset_zip_path = self.download_dir / "Vaihingen" / "ISPRS_semantic_labeling_Vaihingen.zip"
        mask_dataset_zip_path = self.download_dir / "Vaihingen" / "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip"

        extractor.extract_zip_archive(vaihingen_zip_path, self.download_dir, [image_dataset_zip_path.name, mask_dataset_zip_path.name])
        print("Extracted downloaded archives")

        if low_storage_mode:
            #vaihingen_zip_path.unlink()
            print("Deleting downloaded archive to save storage sapce")

        self.d_image_dir.mkdir(exist_ok=True, parents=True)
        self.d_mask_dir.mkdir(exist_ok=True, parents=True)

        extractor.extract_zip_archive(image_dataset_zip_path, image_dataset_zip_path.parent, ["top"])
        extractor.extract_zip_archive(mask_dataset_zip_path, self.d_mask_dir)
        print("Extracted downloaded dataset")

        if low_storage_mode:
            #image_dataset_zip_path.unlink()
            #mask_dataset_zip_path.unlink()
            print("Deleted extracted archives")

        for image_path in (image_dataset_zip_path.parent / "top").glob("*.tif"):
            shutil.move(image_path, self.d_image_dir)
        print("Moved extracted images and masks")

        if low_storage_mode:
            #shutil.rmtree(image_dataset_zip_path.parent)
            pass
    
    def _extract_potsdam(self, extractor, low_storage_mode: bool):
        potsdam_zip_path = self.download_dir / "potsdam.zip"
        images_zip_path = self.download_dir / "Potsdam" / "2_Ortho_RGB.zip"
        masks_zip_path = self.download_dir / "Potsdam" / "5_Labels_all.zip"

        extractor.extract_zip_archive(potsdam_zip_path, self.download_dir, [images_zip_path.name, masks_zip_path.name])
        if low_storage_mode:
            #potsdam_zip_path.unlink()
            print("Deleted downloaded archive")

        images_temp_dir = images_zip_path.parent / images_zip_path.stem
        extractor.extract_zip_archive(images_zip_path, images_zip_path.parent)

        masks_temp_dir = masks_zip_path.parent / masks_zip_path.stem
        (masks_temp_dir).mkdir(exist_ok=True)
        extractor.extract_zip_archive(masks_zip_path, masks_temp_dir)

        print("Extracted images and masks sub archives")

        if low_storage_mode:
            #images_zip_path.unlink()
            #masks_zip_path.unlink()
            print("Deleted extracted sub archives")

        self.d_image_dir.mkdir(exist_ok=True, parents=True)
        self.d_mask_dir.mkdir(exist_ok=True, parents=True)
        #Copy Images, Masks to Correct Directories
        for image_path in images_temp_dir.glob("*.tif"):
            shutil.move(image_path, self.d_image_dir / image_path.name.replace("_RGB", ""))
        for mask_path in masks_temp_dir.glob("*.tif"):
            shutil.move(mask_path, self.d_mask_dir / mask_path.name.replace("_label", ""))
        print("Moved extracted images and masks")

        if low_storage_mode:
            #shutil.rmtree(images_zip_path.parent)
            print("Cleanup directory structure")

    def _get_file_region(self, file_name):
        if "mosaic" in file_name:
            return "vaihingen"
        elif "potsdam" in file_name:
            return "potsdam"

    def upload_train_dataset(self, storage: ContaboStorage):
        train_files_list = list(self.mask_dir.glob("*.jpg")) 
        for file_path in tqdm(train_files_list):
            storage.upload_train_pair(file_path.name, self.image_dir, self.mask_dir)

    def upload_test_dataset(self, storage: ContaboStorage):
        test_files_list = list(self.mask_dir.glob("*.tif"))
        for file_path in tqdm(test_files_list):
            storage.upload_test_pair(file_path.name, self.t_image_dir, self.t_mask_dir)

    def catalog_train_dataset(self):
        """Returns a DataFrame with training patch names and dataset name as columns"""

        df = pd.DataFrame({"name": [x.name for x in self.image_dir.rglob("*.jpg")]}) 
        df["dataset"] = [self.root_dir.name]*len(df)
        return df

    def update_train_dataset(self, storage: ContaboStorage):

        #Set path for catalog to be downloaded to 
        #WARINING: training catalog path must be hamed "patches.csv"
        catalog_path = self.root_dir / "metadata" / "patches.csv"

        #If catalog file present in local storage, remove it
        catalog_path.unlink(missing_ok=True)
        catalog_path.parent.mkdir(exist_ok=True, parents=True)

        #Download catalog from bucket
        storage.download_train_catalog(catalog_path)

        #If a catalog is present in storage 
        if catalog_path.exists() and catalog_path.is_file():
            print("Downloaded Catalog Found")
            df = pd.read_csv(catalog_path)

            #Get a view of files associated with current dataset
            dataset_df = df[df["dataset"] == self.root_dir.name]

            #If any files in bucket are associated with the current dataset
            if len(dataset_df) > 0:
               #Remove those files from storage
                print(f"Deleting existing patches in bucket")
                existing_file_names = list(dataset_df.name.apply(lambda x: x))
                for file_name in tqdm(existing_file_names):
                    storage.delete_train_pair(file_name)
                
                #Remove those files from the catalog
                df = df.drop(dataset_df.index, axis = 0)

        
        #If catalog is not present in storage => dataset is not present in storage
        else:  
            #Create a new catalog placeholder
            print("Downloaded Catalog Not Found, Creating New")
            df = pd.DataFrame({"name" : list(), "dataset" : list()})
        
        #Upload training dataset to bucket
        print("Uploading patches to bucket")
        self.upload_train_dataset(storage)
        #Add patches in image, mask dir to catalog
        df = pd.concat([df, self.catalog_train_dataset()], axis = 0)
        #Save catalog to local storage
        df.to_csv(catalog_path, index = False)
        #Upload catalog to bucket
        storage.upload_train_catalog(catalog_path)
        catalog_path.unlink()