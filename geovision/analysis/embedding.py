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

# NOTE Objectives:
# 1. Run inference on the model to store encoder, pre-final layer and output embeddings (+ metrics in-case of localization/segmentation problems)
# 2. Read embeddings from the file (as a dataframe ?) -> join with dataset metadata -> analyze embeddings to find suspicious samples 
# 3. Visualize embeddings space using t-SNE / UMAP

# TODO:
# 1. impl. core functionality first, discover any useful modificiations to the workflow, focus on integration later.
# 2. each analysis submodule works independent of others, but on the same configuration format and data layout format

# 3. get dataset, dataloader and model config from config.yaml
# 4.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.encoder: torch.nn.Module = model_config.encoder_constructor(**model_config.encoder_params)
        self.decoder: torch.nn.Module = model_config.decoder_constructor(**model_config.decoder_params)
        # self.num_decoder_layers = len(list(self.decoder.children()))

        if model_config.ckpt_path is not None:
            print(f"loading ckpt: {model_config.ckpt_path}")
            weights = torch.load(model_config.ckpt_path, weights_only=True)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            self.load_state_dict(weights)

    def forward(self, inputs: torch.Tensor):
        return self.decoder(self.encoder(inputs))

    #def prefinal(self, inputs: torch.Tensor):
        #inputs = self.encoder(inputs)
        #if self.num_decoder_layers == 0:
            #return inputs
        #for layer_idx, layer in enumerate(self.decoder.children()):
            #if layer_idx == self.num_decoder_layers - 1:
                #return inputs
            #inputs = layer(inputs)

def init_inference(config: ExperimentConfig):
    torch.manual_seed(config.random_seed) 
    np.random.seed(config.random_seed)
    dataset = config.dataset_constructor(split = "all", config = config.dataset_config)
    dataloader = DataLoader(dataset, **config.dataloader_config.params) 
    model = ClassificationModel(config.model_config).to(DEVICE)
    model.eval()
    return dataset, dataloader, model

if __name__ == "__main__":
    cli = argparse.ArgumentParser("model analysis: embedding space")
    cli.add_argument("--path_to_config", '-p', help = "path to config (.yaml) file")
    cli.add_argument("--embedding", '-e', help = "embeddings to compute", choices = ["encoder", "output"])
    cli.add_argument("--metrics", '-m', help = "compute metrics", action = "store_true")
    args = cli.parse_args()

    assert Path(args.path_to_config).is_file(), f"config error (invalid path), expected path to valid config file, got [{args.path_to_config}]"
    config = ExperimentConfig.from_yaml(args.path_to_config)
    dataset, dataloader, model = init_inference(config)

    if args.embedding == "encoder":
        embedding_shape = get_embedding_shape(model.encoder, dataset)

        # if args.write_as_jpg:
            # # apply img2jpg across the 0th dim 
            # out_shape = (len(dataset.split_df),)
            # out_dtype = np.float16
        # else:
            # input_sample, target_sample, _ = dataset[0]
            # input_sample = input_sample.unsqueeze(0).to(DEVICE)
            # input_sample.requires_grad_()

            # target_sample = torch.tensor(target_sample).unsqueeze(0).to(DEVICE)
            # out_shape = attr_module.attribute(inputs = input_sample, target = target_sample).squeeze().permute(1,2,0).detach().cpu().numpy().shape
            # out_shape = (len(dataset.split_df), *out_shape)
            # out_dtype = h5py.special_dtype(vlen = np.uint8)
        
        logs = config.experiments_dir / f"{config.model_config.encoder_name}_{config.model_config.decoder_name}_embeddings.h5"
        with h5py.File(logs, mode = 'a') as logfile: 
            group_name = f"{args.embedding}_embeddings"

            if logfile.get(group_name) is not None:
                print(f"{group_name} already exists in {logs}, overwriting")
                del logfile[group_name]

            group = logfile.create_group(group_name)

            # group.attrs["image_transforms"] = str(config.dataset_config.image_pre)
            # for key, value in config.model_config.__dict__.items():
                # if "name" in key:
                    # group.attrs.modify(key, value)
                # elif "params" in key:
                    # for param in value:
                        # group.attrs.modify(f"{key}_{param}", value[param])
            # group.create_dataset("indices", shape = out_shape, dtype = np.uint32)
            # group.create_dataset("maps", shape = out_shape, dtype = out_dtype)

            # if attr_params.get("return_convergence_delta", False):
                # group.create_dataset("deltas", shape = len(dataset.split_df), dtype = np.float32)

            # for batch_idx, batch in tqdm(enumerate(iter(dataloader)), total = len(dataloader)):
                # maps = attr_module.attribute(inputs = batch[0].to(DEVICE), target = batch[1].to(DEVICE), **attr_params)
                # if isinstance(maps, tuple):
                    # maps, deltas = maps
                # else:
                    # deltas = None
                # maps = maps.detach().cpu().permute(0,2,3,1).squeeze().numpy()

                # if args.save_as_jpg:
                    # maps = maps.apply(img2jpg)

                # start, end = len(batch[2]) * batch_idx, len(batch[2]) * (batch_idx + 1)
                # group["indices"][start:end] = batch[2]
                # group["maps"][start:end] = maps 
                # if deltas is not None: 
                    # group["deltas"][start:end] = deltas.detach().cpu().squeeze().numpy()
        
    # elif args.operation == "draw":
        # assert Path(args.path_to_config).is_file(), f"config error (invalid path), expected path to valid config file, got [{args.path_to_config}]"
        # assert Path(args.path_to_attribution).expanduser().is_file(), f"config error (invalid path), expected path to valid attribution file, got [{args.path_to_attribution}]"
        # assert args.visualization in ("image_overlay", "channel_overlay", "adjacent_plots")
        # config = ExperimentConfig.from_yaml(args.path_to_config)
        # dataset = init_dataset(config)

        # if args.visualization == "image_overlay":
            # plotting_fn = plot_image_overlay
        # elif args.visualization == "channel_overlay":
            # plotting_fn = plot_channel_overlay
        # else:
            # plotting_fn = plot_adjacent_plots

        # def get_attr_name(config_yaml):
            # with open(config_yaml) as f:
                # return yaml.load(f, Loader=yaml.SafeLoader)["attribution"]

        # def plot_and_save(idx: int):
            # image, target, df_idx = dataset[idx]
            # image = image.permute(1,2,0).numpy()

            # attr_map = group["maps"][indices.index(df_idx)]
            # if attr_map.ndim == 1:
                # attr_map = imread(BytesIO(attr_map), extension=".jpg")
            
            # fig, ax = plt.subplots(1,1, figsize = (10,10), layout = "constrained")
            # plotting_fn(image, attr_map, target, df_idx)
            # #print(attr_map.shape, attr_map.ndim, image.shape)
        
        # # def process_one_batch(batch) -> pool.map_unordered(process_one_batch, dataloader)
        # with h5py.File(args.path_to_attribution, mode = 'r') as logfile:
            # group = logfile[f"{get_attr_name(args.path_to_config)}"]
            # indices = group["indices"][:].tolist()

            # with mp.Pool(mp.cpu_count()) as pool:
                # pool.map(plot_and_save, dataset.split_df.iloc[:10].index)
    # else:
        # raise AssertionError(f"config error (invalid value), expected :operation to be one of compute or draw, got {args.operation}")