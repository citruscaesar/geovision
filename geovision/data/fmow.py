import PIL
import json
import tarfile
import numpy as np
import pandas as pd
import pandera as pa
from tqdm import tqdm
from pathlib import Path
import imageio.v3 as iio
from .dataset import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from geovision.io.local import get_new_dir, get_valid_file_err, is_empty_dir
from skimage.transform import resize

class FMoW:
    local = Path.home()/"datasets"/"fmow-rgb"
    cold = Path("/run/media/sambhav/StorageHDD_1/datasets/fmow-rgb/") 
    # fmt: off
    class_names = (
        'airport', 'airport_hangar', 'airport_terminal', 'amusement_park',
        'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint',
        'burial_site', 'car_dealership', 'construction_site', 'crop_field',
        'dam', 'debris_or_rubble', 'educational_institution',
        'electric_substation', 'factory_or_powerplant', 'false_detection',
        'fire_station','flooded_road', 'fountain', 'gas_station', 'golf_course',
        'ground_transportation_station', 'helipad', 'hospital',
        'impoverished_settlement', 'interchange', 'lake_or_pond',
        'lighthouse', 'military_facility', 'multi-unit_residential',
        'nuclear_powerplant', 'office_building', 'oil_or_gas_facility',
        'park', 'parking_lot_or_garage', 'place_of_worship',
        'police_station', 'port', 'prison', 'race_track', 'railway_bridge',
        'recreational_facility', 'road_bridge', 'runway', 'shipyard',
        'shopping_mall', 'single-unit_residential', 'smokestack',
        'solar_farm', 'space_facility', 'stadium', 'storage_tank',
        'surface_mine', 'swimming_pool', 'toll_booth', 'tower',
        'tunnel_opening', 'waste_disposal', 'water_treatment_facility',
        'wind_farm', 'zoo'
    )
    large_image_classes = (
        "airport", "amusement_park", "impoverished_settlement", 
        "nuclear_powerplant", "port", "runway", "shipyard", "space_facility"
    )
    # dataframe_schema = pa.DataFrameSchema({
        # "image_path": pa.Column(str, coerce=True),
        # "label_str": pa.Column(str),
        # "height": pa.Column(int, coerce=True),
        # "width": pa.Column(int, coerce=True),
        # "bbox": pa.Column(list, coerce=True)
    # })

    crs = "EPSG:4326"
    # fmt: on

    # FMoW Challenges
    # Some images contain more than one class, making it a multilabel classification problem
    # Some classes have very large images, with out of context information, which need to be smartly cropped 
    #
    # iterate through the bounding boxes and gather all labels in the dataset_df
    # for multiclass classification,
    #   if bbox.larger_side > 2000px: crop a square of side=larger_side with the bbox at the center and resize to 2000px 
    #   elif bbox.larger_side <= 2000px: crop upto 2000px
    # for multilabel classification,
    #   calculate superset bbox for each image, use the same logic as multiclass to crop
    # put images and metadata into hdf5, dynamically crop during runtime? or crop first according to task and then make hdf

    @classmethod
    def extract_metadata_files_to_tempfolder(cls):
        temp_gt = get_new_dir(cls.local/"gt")
        with tarfile.open(cls.cold/"groundtruth.tar.bz2") as gt:
            gt.extractall(temp_gt)

    @classmethod
    def get_dataset_df_from_archive(cls):
        # TODO: add geometry conversion to convert polygon to affine matrix 
        # TODO: add mapping for seq and test using the .jsons in gt 

        metadata_dir = get_new_dir(cls.local/"gt")
        try:
            if is_empty_dir(metadata_dir):
                with tarfile.open(get_valid_file_err(cls.cold/"groundtruth.tar.bz2")) as archive:
                    archive.extractall(metadata_dir)
            metadata_df = pd.read_hdf(cls.local/"gt"/"metadata.h5", key = "dataset_df", mode = 'r')
        except OSError:
            metadata_dict = {k:list() for k in ("image_path", "subdir", "label_str", "height", "width", "bbox", "gsd", "utm", "loc", "timestamp")}
            for json_path in tqdm(metadata_dir.rglob("*_rgb.json"), total = 523846):
                with open(json_path) as json_file:
                    json_parsed = json.load(json_file)
                for box in json_parsed["bounding_boxes"][1:]:
                    image_path = Path('/'.join(str(json_path).split('/')[-4:-1]))/json_parsed["img_filename"]
                    metadata_dict["image_path"].append(image_path)
                    metadata_dict["subdir"].append(image_path.parent)
                    metadata_dict["label_str"].append(box["category"])
                    metadata_dict["height"].append(json_parsed["img_height"])
                    metadata_dict["width"].append(json_parsed["img_width"])
                    metadata_dict["bbox"].append(box["box"])
                    metadata_dict["gsd"].append(json_parsed["gsd"])
                    metadata_dict["utm"].append(json_parsed["utm"])

                    metadata_dict["loc"].append(json_parsed["raw_location"])
                    metadata_dict["timestamp"].append(json_parsed["timestamp"])
            
            metadata_df = (
                pd.DataFrame(metadata_dict)
                .assign(image_path = lambda df: df["image_path"].apply(lambda x: str(x).replace("_gt", "")))
                .assign(subdir = lambda df: df["image_path"].apply(lambda x: str((Path(x).parent))))
                .sort_values(["subdir", "timestamp"])
                .drop(columns = "subdir")
                .reset_index(drop = True)
            )
            metadata_df.to_hdf(cls.local/"gt"/"metadata.h5", key = "dataset_df", mode = 'w', format = "fixed")
        return (
            metadata_df
            .assign(image_path = lambda df: df["image_path"].apply(lambda x: Path(x)))
            .assign(bbox = lambda df: df["bbox"].astype(object))
            .assign(timestamp = lambda df: pd.to_datetime(df["timestamp"], format='mixed'))
        )

    @classmethod
    def get_mcc_dataset_df(cls) -> pd.DataFrame:
        # TODO: false detection is pointless in mcc case, remove it
        # TODO: only keep the label bbox params as 4 seperate columns 
        # TODO: merge seq and test into one? 

        def get_corners_from_bbox(bbox: tuple[int, int, int, int]):
            #           bbox                     corners
            #       (x, y, w, h)                (tl, br)
            #  (x,y) -----w-------|      (y,x) -------------|
            #    |                |        |                |
            #    |                |        |                |
            #    h                |   ->   |                |
            #    |                |        |                |
            #    |----------------|        | -------------(y,x)
            return [bbox[1], bbox[0]], [bbox[1] + bbox[3], bbox[0] + bbox[2]]
        
        def get_crop_corners(row: pd.Series):

            # TODO: need to refactor and document this and add utils like def shift origin
            # TODO: this also needs to transform the affine matrix for the transform 

            image_dims = (row["height"], row["width"])
            label_tl, label_br = get_corners_from_bbox(row["bbox"])
            crop_tl, crop_br = get_corners_from_bbox(row["bbox"])

            label_height, label_width = label_br[1] - label_tl[1], label_br[0] - label_tl[0]
            if label_height > label_width:
                ldim, sdim = 1, 0
            else:
                ldim, sdim = 0, 1
            diff = abs(label_height - label_width)

            # if the length of the image / bbox along the larger dim is smaller than 2000px, ignore it
            if image_dims[ldim] <= 2000: # or label_br[ldim] - label_tl[ldim] <= 2000:
                return (0,0), image_dims, label_tl, label_br, (sdim, 0, 0)

            # evenly divide the difference to be distributed along the smaller dim
            pre, post = 2*(diff//2,)
            pre += 1 if diff % 2 == 0 else 0
            # if the bbox can be extended along the smaller dim 
            # if image height/width >    
            # if image_dims[sdim] > (label_br[ldim] - label_tl[ldim]):
            # if there isn't enough space before the bbox along the smaller dim
            if label_tl[sdim] < pre:
                # adjust the remaining length after the bbox
                post = post + pre - label_tl[sdim]
                # use all the space available before the bbox
                pre = label_tl[sdim]
            # if there isn't enough space after the bbox along the smaller dim
            elif (rem := image_dims[sdim] - label_br[sdim]) < post:
                # adjust the remaining length before the bbox 
                pre = pre + post - rem 
                # use all the space available after the bbox 
                post = rem 

            crop_tl[sdim] -= pre
            crop_br[sdim] += post
            label_tl[sdim] += pre
            label_br[sdim] += pre
            label_br[ldim] -= label_tl[ldim]
            label_tl[ldim] = 0

            # padding if needed along the smaller dim
            pre_pad, post_pad = 0, 0
            if crop_tl[sdim] < 0:
                pre_pad = abs(crop_tl[sdim])
                crop_tl[sdim] += pre_pad
                crop_br[sdim] += pre_pad
            if crop_br[sdim] > image_dims[sdim]:
                post_pad = abs(crop_br[sdim] - image_dims[sdim])

            return tuple(crop_tl), tuple(crop_br), tuple(label_tl), tuple(label_br), (sdim, pre_pad, post_pad)

        df = cls.get_dataset_df_from_archive()
        df = df[df["image_path"].apply(lambda x: "airport" in str(x))]
        df = df[df["image_path"].apply(lambda x: "seq" not in str(x))]
        df = df[df["image_path"].apply(lambda x: "test" not in str(x))]
        df[["crop_tl", "crop_br", "label_tl", "label_br", "pad_info"]] = df.apply(lambda x: get_crop_corners(x), axis = 1, result_type = "expand")
        return df
    
    @classmethod
    def plot_mcc_sample(cls, row: pd.Series):
        def get_rectangle_from_corners(tl: tuple[int, int], br: tuple[int, int], **kwargs) -> Rectangle:
            #          corners                   rect 
            #         (tl, br)                ((x,y), w, h)
            #   (y,x) ------------|        |----------------|
            #    |                |        |                |
            #    |                |        h                |
            #    |                |   ->   |                |
            #    |                |        |                |
            #    |--------------(y,x)      (x,y) -----w-----|
            return Rectangle((tl[1], br[0]), br[1] - tl[1], tl[0] - br[0], **kwargs)

        PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        image = iio.imread(cls.local/row["image_path"])
        crop_tl, crop_br = row["crop_tl"], row["crop_br"]

        dim, pre, post = row["pad_info"]
        if pre or post:
            if dim == 0:
                image = np.pad(image, ((pre, post), (0, 0), (0, 0)))    
            else:
                image = np.pad(image, ((0, 0), (pre, post), (0, 0)))

        cropped_image = image[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1], :]
        crop_rect = get_rectangle_from_corners(crop_tl, crop_br, linewidth=2, edgecolor='b', facecolor='none')
        #label_rect = get_rectangle_from_corners(row["label_tl"], row["label_br"], linewidth=2, edgecolor='r', facecolor='none')

        fig = plt.figure(figsize = (6, 3), layout = "tight")
        l = plt.subplot(121)
        r = plt.subplot(122)
        l.imshow(image)
        l.add_patch(crop_rect)
        r.imshow(cropped_image)
        #r.add_patch(label_rect)
        #ax.axis("off")
        fig.savefig(f'fmow/{row["image_path"].stem}.png')
        plt.cla()
        fig.clf()
        plt.close('all')
    
    @classmethod
    def transform_to_mcc_imagefolder(cls, row: pd.Series):
        #imagefolder_path = get_new_dir(cls.local/"imagefolder")
        #df = cls.get_mcc_dataset_df_from_imagefolder()

        PIL.Image.MAX_IMAGE_PIXELS = 20000 * 20000 * 3
        image = iio.imread(cls.local/row["image_path"])
        crop_tl, crop_br = row["crop_tl"], row["crop_br"]

        dim, pre, post = row["pad_info"]
        if pre or post:
            if dim == 0:
                image = np.pad(image, ((pre, post), (0, 0), (0, 0)))    
            else:
                image = np.pad(image, ((0, 0), (pre, post), (0, 0)))
        image = image[crop_tl[0]:crop_br[0], crop_tl[1]:crop_br[1], :]
        image = resize(image, (2000, 2000, 3), preserve_range=True, anti_aliasing=False)
        image = image.astype(np.uint8)
        iio.imwrite(get_new_dir(cls.temp_local/"imagefolder"/row["image_path"].parent)/row["image_path"].name, image, extension = ".jpg")
        del image
    
    @classmethod
    def transform_to_mcc_hdf5(cls):
        pass
    

class FMoWImagefolderClassification(Dataset):
    def __init__(self):
        pass
