def transformation_strategy_cityosm(dataset_zip_path: Path, image_dir: Path, mask_dir: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path) as zf:
        filenames = [x.removesuffix("_image.png") for x in zf.namelist()[1:] if "_image" in x]
        for filename in tqdm(sorted(filenames), desc = f"{dataset_zip_path.stem.capitalize()} Progress"):
            image_src_path = f"{filename}_image.png"
            image_dst_path = image_dir/f"{filename.split('/')[-1]}.png"

            mask_src_path = zipfile.Path(dataset_zip_path) / f"{filename}_labels.png"
            mask_dst_path = mask_dir/f"{filename.split('/')[-1]}.png"

            extract_image(image_src_path, image_dst_path, zf) 

            # Mask[:, :, 0] = Road
            # Mask[:, :, 1] = Building and Road
            # Mask[:, :, 2] = Building 
            mask = iio.imread(str(mask_src_path), extension=".png")
            mask = mask[:, :, 2]
            mask = np.where(mask==255, 0, 255).astype(np.uint8)
            iio.imwrite(mask_dst_path, mask, extension=".png")

class CityOSMETL():
    urls = {
        "berlin.zip": "https://zenodo.org/record/1154821/files/berlin.zip?download=1",
        "chicago.zip": "https://zenodo.org/record/1154821/files/chicago.zip?download=1",
        "paris.zip": "https://zenodo.org/record/1154821/files/paris.zip?download=1",
        "potsdam.zip": "https://zenodo.org/record/1154821/files/potsdam.zip?download=1",
        "tokyo.zip": "https://zenodo.org/record/1154821/files/tokyo.zip?download=1",
        "zurich.zip": "https://zenodo.org/record/1154821/files/zurich.zip?download=1"
    } 

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
        asyncio.run(downloader.download_files())
    
    def _move_and_rename_files(self, file_paths) -> None:
        for file_path in file_paths:
            shutil.move(file_path, file_path.parents[1] / f"{file_path.stem.split('_')[0]}.png")
            
    def _extract_file_num(self, file_name: str):
        numbers = list()
        for char in file_name:
            if char.isdigit():
                numbers.append(char) 
        return int(''.join(numbers))   

    def extract(self, low_storage_space:bool):
        extractor = DatasetExtractor()

        print(f"Extracting downloaded archives")
        for zip_file_name in tqdm(self.urls.keys()):
            zip_file_path = self.download_dir / zip_file_name
            extractor.extract_zip_archive(zip_file_path, self.d_image_dir, ["image"])
            extractor.extract_zip_archive(zip_file_path, self.d_mask_dir, ["labels"])

        print(f"Reorganizing file structure")
        self._move_and_rename_files(self.d_image_dir.rglob("*.png"))
        self._move_and_rename_files(self.d_mask_dir.rglob("*.png"))

        if low_storage_space: 
            print(f"Deleting downloaded archives to save storage space")
            for zip_file_name in self.urls.keys():
                zip_file_path = self.download_dir / zip_file_name
                zip_file_path.unlink()
    
    def calculate_train_test_split(self):
        self.files_list = [x.name for x in self.d_image_dir.rglob("*.png")]
        location_numbers = dict()
        for file_name in self.files_list:
            location = file_name.split('.')[0].strip('1234567890')
            if location in location_numbers.keys():
                location_numbers[location].append(self._extract_file_num(file_name))
            else:
                location_numbers[location] = list()

        self.test_files_list = list()
        for location, numbers_list in location_numbers.items():
            for num in sorted(numbers_list)[:int(0.15*len(numbers_list))]:
                self.test_files_list.append(f"{location}{num}.png")
        
        self.train_files_list = list(set(self.files_list).difference(self.test_files_list))

    def move_test_split(self):
        self.t_image_dir.mkdir(exist_ok=True, parents=True)
        self.t_mask_dir.mkdir(exist_ok=True, parents=True)

        for file_name in self.test_files_list:
            shutil.move(self.d_image_dir / file_name, self.t_image_dir)
            shutil.move(self.d_mask_dir / file_name, self.t_mask_dir)
    
    @staticmethod
    def read_mask(path: Path):
        mask = skimage.io.imread(path)[:, :, 2] # type: ignore
        mask = np.where((mask == 0), np.uint8(255), np.uint8(0))
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
    
    def delete_downloads(self):
        shutil.rmtree(self.download_dir)

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