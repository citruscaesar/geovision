from collections.abc import Callable
from numpy.typing import NDArray 
from typing import Any

import yaml
import torch
import h5py
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from imageio.v3 import imread, imwrite
from torch.utils.data import DataLoader
from geovision.experiment.config import ExperimentConfig
from geovision.models.interfaces import ModelConfig
from captum.attr import (
    Attribution,
    IntegratedGradients,
    DeepLift
)

# TODO:
# 1. [.] Impl. all attribution methods, including layer and neuron attribution
# 2. [x] Compress attribution maps to JPEG -> even float16 takes too much space

# 3. [.] Impl. visualizing grids -> image_overlay, channel_overlay, image_attrib_grid -> save_to_dir
#       -> must include model prediction along with actual target
#       -> plot attribution w.r.t. both actual label and model prediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.encoder = model_config.encoder_constructor(**model_config.encoder_params)
        self.decoder = model_config.decoder_constructor(**model_config.decoder_params)

        if model_config.ckpt_path is not None:
            print(f"loading ckpt: {model_config.ckpt_path}")
            weights = torch.load(model_config.ckpt_path, weights_only=True)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            self.load_state_dict(weights)

    def forward(self, inputs: torch.Tensor):
        return self.decoder(self.encoder(inputs))

def get_attribution(model: torch.nn.Module, config_path: Path) -> tuple[Attribution, dict[str, Any]]:
    attribution_modules: dict[str, Callable[..., Attribution]] = {
        "integrated_gradients": IntegratedGradients, 
        "deep_lift": DeepLift
    }
    with open(config_path) as f:
        params = yaml.load(f, Loader = yaml.SafeLoader)

    assert params["attribution"] in attribution_modules, \
        f"config error (invalid value), expected :attribution to be one of {attribution_modules}, got {params["attribution"]}"

    assert isinstance(params["attribution_init_params"], dict), \
        f"config error (invalid type), expected :attribution_init_params to be dict, got {type(params["attribution_init_params"])}"

    assert isinstance(params["attribution_fn_params"], dict), \
        f"config error (invalid type), expected :attribution_fn_params to be dict, got {type(params["attribution_fn_params"])}"

    return (
        params["attribution"], 
        attribution_modules[params["attribution"]](model, **params["attribution_init_params"]), 
        params["attribution_fn_params"]
    )
    
def init_inference(config: ExperimentConfig):
    torch.manual_seed(config.random_seed) 
    np.random.seed(config.random_seed)
    dataset = config.dataset_constructor(split = "all", config = config.dataset_config)
    dataloader = DataLoader(dataset, **config.dataloader_config.params) 
    model = ClassificationModel(config.model_config).to(DEVICE)
    model.eval()
    return dataset, dataloader, model

def init_dataset(config: ExperimentConfig):
    torch.manual_seed(config.random_seed) 
    np.random.seed(config.random_seed)
    dataset = config.dataset_constructor(split = "all", config = config.dataset_config)
    return dataset

def get_input_shape(dataset):
    return dataset[0][0].permute(1,2,0).shape

def plot_image_overlay(fig: Figure, ax: Axes, image: NDArray, attr_map: NDArray, label_idx: int, pred_label_idx: int, df_idx: int):
    pass

def plot_channel_overlay():
    pass

def plot_adjacent_plots():
    pass

@np.vectorize
def img2jpg(image: NDArray) -> NDArray:
    image = image - image.min() / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    image = imwrite('<bytes>', image, extension=".jpg")
    image = np.frombuffer(image, dtype = np.uint8)
    return image 

if __name__ == "__main__":
    cli = argparse.ArgumentParser("model analysis: attribution methods")
    cli.add_argument("--path_to_config", '-p', help = "path to config (.yaml) file")
    cli.add_argument("--path_to_attribution", '-a', help = "path to attribution (.h5) file")
    cli.add_argument("--operation", '-o', help = "operation to perform", choices=["compute", "draw"])
    cli.add_argument("--visualization", '-v', help = "kind of plots to draw", choices=["image_overlay", "channel_overlay", "adjacent_plot"])
    cli.add_argument("--write_as_jpg", help = "compress attribution maps as uint8 jpg", action = "store_true")
    args = cli.parse_args()

    # 1. Use the dataset to calculate and store attribution maps 
    #   1. Config File to init the dataloader, model and attribution method 
    #   2. (Optional) Attribution File
    #   3. Arg to specify to store as fp16 or jpeg
    # 2. Use the dataset and stored attribution maps to draw plots and store somewhere 
    #   1. Config File to init the dataset 
    #   2. Attribution File for the saved attributions 
    #   3. Argument to specify what kind of plots to draw

    if args.operation == "compute":
        assert Path(args.path_to_config).is_file(), f"config error (invalid path), expected path to valid config file, got [{args.path_to_config}]"
        config = ExperimentConfig.from_yaml(args.path_to_config)
                
        dataset, dataloader, model = init_inference(config)
        attr_name, attr_module, attr_params = get_attribution(model, args.path_to_config)

        if args.write_as_jpg:
            # apply img2jpg across the 0th dim 
            out_shape = (len(dataset.split_df),)
            out_dtype = np.float16
        else:
            input_sample, target_sample, _ = dataset[0]
            input_sample = input_sample.unsqueeze(0).to(DEVICE)
            input_sample.requires_grad_()

            target_sample = torch.tensor(target_sample).unsqueeze(0).to(DEVICE)
            out_shape = attr_module.attribute(inputs = input_sample, target = target_sample).squeeze().permute(1,2,0).detach().cpu().numpy().shape
            out_shape = (len(dataset.split_df), *out_shape)
            out_dtype = h5py.special_dtype(vlen = np.uint8)
        
        logs = config.experiments_dir / f"{config.dataset_name.split('_')[0]}_{config.model_config.encoder_name}_attribution.h5"
        with h5py.File(logs, mode = 'a') as logfile: 
            if logfile.get(attr_name) is not None:
                print(f"{attr_name} already exists in {logs}, overwriting")
                del logfile[attr_name]

            group = logfile.create_group(attr_name)
            group.attrs["image_transforms"] = str(config.dataset_config.image_pre)
            for key, value in config.model_config.__dict__.items():
                if "name" in key:
                    group.attrs.modify(key, value)
                elif "params" in key:
                    for param in value:
                        group.attrs.modify(f"{key}_{param}", value[param])
            group.create_dataset("indices", shape = out_shape, dtype = np.uint32)
            group.create_dataset("maps", shape = out_shape, dtype = out_dtype)

            if attr_params.get("return_convergence_delta", False):
                group.create_dataset("deltas", shape = len(dataset.split_df), dtype = np.float32)

            for batch_idx, batch in tqdm(enumerate(iter(dataloader)), total = len(dataloader)):
                maps = attr_module.attribute(inputs = batch[0].to(DEVICE), target = batch[1].to(DEVICE), **attr_params)
                if isinstance(maps, tuple):
                    maps, deltas = maps
                else:
                    deltas = None
                maps = maps.detach().cpu().permute(0,2,3,1).squeeze().numpy()

                if args.save_as_jpg:
                    maps = maps.apply(img2jpg)

                start, end = len(batch[2]) * batch_idx, len(batch[2]) * (batch_idx + 1)
                group["indices"][start:end] = batch[2]
                group["maps"][start:end] = maps 
                if deltas is not None: 
                    group["deltas"][start:end] = deltas.detach().cpu().squeeze().numpy()
        
    elif args.operation == "draw":
        assert Path(args.path_to_config).is_file(), f"config error (invalid path), expected path to valid config file, got [{args.path_to_config}]"
        assert Path(args.path_to_attribution).expanduser().is_file(), f"config error (invalid path), expected path to valid attribution file, got [{args.path_to_attribution}]"
        assert args.visualization in ("image_overlay", "channel_overlay", "adjacent_plots")
        config = ExperimentConfig.from_yaml(args.path_to_config)
        dataset = init_dataset(config)

        if args.visualization == "image_overlay":
            plotting_fn = plot_image_overlay
        elif args.visualization == "channel_overlay":
            plotting_fn = plot_channel_overlay
        else:
            plotting_fn = plot_adjacent_plots

        def get_attr_name(config_yaml):
            with open(config_yaml) as f:
                return yaml.load(f, Loader=yaml.SafeLoader)["attribution"]

        def plot_and_save(idx: int):
            image, target, df_idx = dataset[idx]
            image = image.permute(1,2,0).numpy()

            attr_map = group["maps"][indices.index(df_idx)]
            if attr_map.ndim == 1:
                attr_map = imread(BytesIO(attr_map), extension=".jpg")
            
            fig, ax = plt.subplots(1,1, figsize = (10,10), layout = "constrained")
            plotting_fn(image, attr_map, target, df_idx)
            #print(attr_map.shape, attr_map.ndim, image.shape)
        
        # def process_one_batch(batch) -> pool.map_unordered(process_one_batch, dataloader)
        with h5py.File(args.path_to_attribution, mode = 'r') as logfile:
            group = logfile[f"{get_attr_name(args.path_to_config)}"]
            indices = group["indices"][:].tolist()

            with mp.Pool(mp.cpu_count()) as pool:
                pool.map(plot_and_save, dataset.split_df.iloc[:10].index)
    else:
        raise AssertionError(f"config error (invalid value), expected :operation to be one of compute or draw, got {args.operation}")