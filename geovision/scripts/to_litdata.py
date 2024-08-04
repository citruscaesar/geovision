from typing import Callable

import PIL
import yaml
import argparse
import pandas as pd
import pandera as pa
from pathlib import Path
from litdata import optimize
from geovision.io.local import get_new_dir
from geovision.data.config import DatasetConfig

def get_classification_sample(idx:int) -> dict:
    row = df.iloc[idx]
    return {
        "image": PIL.Image.open(str(row["image_path"])),
        "label_idx": int(row["label_idx"]),
        "df_idx": int(row["df_idx"])
    }

def get_segmentation_sample(idx:int) -> dict:
    row = df.iloc[idx]
    return {
        "image": PIL.Image.open(str(row["image_path"])),
        "mask": PIL.Image.open(row["mask_path"]),
        "df_idx": int(row["df_idx"])
    }

def get_dataset_name_and_config() -> tuple[str, DatasetConfig]:
    argparser = argparse.ArgumentParser(prog = "Encode Imagefolder Dataset to Litdata", epilog = "\n")
    argparser.add_argument("--config", default = "config.yaml", help = "path to config file", dest = "config_path")
    args = argparser.parse_args()
    #print(args)

    with open(args.config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
        config_dict["dataset_params"]["random_seed"] = config_dict["random_seed"]
    config = DatasetConfig.model_validate(config_dict["dataset_params"])
    #print(str(config).replace(' ', '\n'))

    return config_dict["dataset"], config

def get_dataset_info() -> tuple[pd.DataFrame, Path, Callable]:
    dataset, config = get_dataset_name_and_config()
    name, _, task = dataset.split('_') 
    if task == "classification":
        fn = get_classification_sample
        schema = pa.DataFrameSchema({
                "image_path": pa.Column(str, coerce=True), 
                "label_idx": pa.Column(int, coerce=True),
                "df_idx": pa.Column(int, coerce=True, unique = True)
        })
    elif task == "segmentation":
        fn = get_segmentation_sample
        schema = pa.DataFrameSchema({
                "image_path": pa.Column(str, coerce=True), 
                "mask": pa.Column(str, coerce=True),
                "df_idx": pa.Column(int, coerce=True, unique = True)
        })
    else:
        raise NotImplementedError("unexpected task in dataset name, got {task}")

    if name == "imagenet":
        from geovision.data.imagenet import Imagenet
        return Imagenet.get_dataset_df_from_litdata(config, schema), Imagenet.local / "litdata", fn
    elif name == "imagenette":
        from geovision.data.imagenette import Imagenette
        return Imagenette.get_dataset_df_from_litdata(config, schema), Imagenette.local / "litdata", fn
    else:
        raise NotImplementedError("unexprected dataset in dataset name, got {name}")

df, litdata_dir, fn = get_dataset_info()

if __name__ == "__main__":
    df.to_csv(get_new_dir(litdata_dir) / "dataset.csv", index = False)
    optimize(
        fn = fn,
        inputs = df[df["split"] == "train"].index.tolist(),
        output_dir = str(get_new_dir(litdata_dir / "train")),
        chunk_bytes = "1GB",
        num_workers = 4
    )
    optimize(
        fn = fn,
        inputs = df[df["split"] == "val"].index.tolist(),
        output_dir = str(get_new_dir(litdata_dir / "val")),
        chunk_bytes = "1GB",
        num_workers = 4
    )
    optimize(
        fn = fn,
        inputs = df[df["split"] == "test"].index.tolist(),
        output_dir = str(get_new_dir(litdata_dir / "test")),
        chunk_bytes = "1GB",
        num_workers = 4
    )