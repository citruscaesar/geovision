from typing import Literal, Optional
from numpy.typing import NDArray
from torch import Tensor


import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from affine import Affine
from shapely import Polygon
from rasterio.crs import CRS
from rasterio.features import sieve, shapes
from skimage.measure import find_contours, approximate_polygon
from torchvision.transforms.v2 import Identity

from tqdm import tqdm
from geovision.io.local import FileSystemIO as fs 
from geovision.models.interfaces import ModelConfig
from geovision.experiment.config import ExperimentConfig
from geovision.data.inria import Inria_Building_Segmentation_HDF5

class UNet(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        self.encoder = config.encoder_constructor(**config.encoder_params)
        config.decoder_params["layer_ch"] = self.encoder._out_ch_per_layer 
        config.decoder_params["layer_up"] = self.encoder._downsampling_per_layer
        self.decoder = config.decoder_constructor(**config.decoder_params)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoder_outputs = list() 
        for layer in self.encoder.children():
            images = layer(images)
            encoder_outputs.append(images)
        return self.decoder(*reversed(encoder_outputs))


def polygonize(mask: NDArray, transform: Affine, min_pixels: int = 20, connectivity: Literal[4, 8] = 4, eps: Optional[float] = None) -> list[Polygon]:

    # contours = skimage.measure.find_contours(mask)
    # fig, ax = plt.subplots(1,1, figsize = (10, 10))
    # ax.imshow(mask, cmap = "gray")
    # for contour in contours:
        # contour = skimage.measure.approximate_polygon(contour, 1)
        # ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    # ax.axis("off");

    mask = sieve(mask, size=min_pixels, connectivity=connectivity)
    polygons = list()
    for contour in find_contours(mask, fully_connected="low" if connectivity == 4 else "high"):
        contour = approximate_polygon(contour, tolerance=eps)
        if transform is not None:
            contour = [transform*(vertex[1], vertex[0]) for vertex in contour]
            # vertices = np.matrix(vertices) # vertices = [[y1, x1], [y2, x2], ..., [yn, xn]], shape = (#vertices, 2) [y,x]
            # vertices[:, [0, 1]] = vertices[:, [1, 0]] # vertices = [[x1, y1], [x2, y2], ..., [xn, yn]], shape = (#vertices, 2) [x,y]
            # vertices = np.c_[vertices, np.ones(vertices.shape[0])] # vertices = [[x1, y1, 1], [x2, y2, 1], ..., [xn, yn, 1]], shape = (#vertices, 3) [x,y,1]
            # vertices = np.transpose(vertices) # shape = (3, #vertices) [each vertex is now a column vector]
            # vertices = np.matmul(transform, vertices) # shape = (3, #vertices) []
            # vertices = np.transpose(vertices[:2]) # shape = (#vertices, 2)
        polygon = Polygon(contour)
        if polygon.is_valid:
            polygons.append(polygon)
    return polygons

    # # https://rasterio.readthedocs.io/en/stable/api/rasterio.features.html#rasterio.features.shapes
    # polygons = shapes(mask, connectivity=connectivity, transform=transform)
    # return list(polygons)

@staticmethod
def bounds(image: NDArray, row: pd.Series) -> Polygon: ...

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    cli = argparse.ArgumentParser("inference script for building footprints")
    cli.add_argument("--config", default = "config.yaml", help = "path to config.yaml")
    cli.add_argument("--ckpt", required=True, help = "path to segmentation model checkpoint")
    cli.add_argument("--split", choices=["train", "val", "test"])
    # cli.add_argument("--reg_ckpt", required=True, help = "path to boundary regularizer model checkpoint")
    args = cli.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    config.dataset_config.train_aug = Identity()

    model = UNet(config.model_config)
    model.compile()
    model.load_state_dict(torch.load(args.ckpt, weights_only=True), strict = False)

    ds = Inria_Building_Segmentation_HDF5("train", config.dataset_config)

    shape = ds("train", config.dataset_config)[0][0].permute(1,2,0).shape

    with h5py.File(config.experiments_dir / "outputs.h5", mode = 'w') as f:
        for ds in ("train", "val", "test"):
            seg_preds = f.create_dataset(f"{ds}/preds", (len(datasets[ds]), *shape), "f16")
            reg_preds = f.create_dataset(f"{ds}/preds", (len(datasets[ds]), *shape), "f16") 

            dataloader = torch.utils.data.DataLoader(Inria_Building_Segmentation_HDF5(ds, config.dataset_config), **config.dataloader_config.params())
            for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
                images = batch[0].to(DEVICE)
                # labels = batch[1].to(DEVICE)

                preds = model(images)
                # preds = reg_model(preds, labels)

                preds = preds.argmax(1).detach().cpu().numpy().astype(np.uint8)
                for pred, idx in zip(preds, batch[-1]):
                    seg_preds[idx] = pred