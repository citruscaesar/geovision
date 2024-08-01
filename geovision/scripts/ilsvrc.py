from typing import Callable

import h5py
import wandb
import torch
import argparse
import torchvision
import torchmetrics
import numpy as np
import pandas as pd
import imageio.v3 as iio
import torchvision.transforms.v2 as t

from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ImagenetValidationDataset(Dataset):
    def __init__(self, transform: t.Transform):
        self.root = Path.home() / "datasets" / "imagenet" / "hdf5" / "imagenete_val.h5"
        self.df = pd.read_hdf(self.root, key="df", mode="r") 
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with h5py.File(self.root, "r") as f:
            image = iio.imread(BytesIO(f["images"][idx]))
        image = np.stack((image,) * 3, axis=-1) if image.ndim == 2 else image
        label = self.df.iloc[idx]["label_idx"]
        return self.transform(image), label

class Config:
    name = ""
    num_workers = 2 
    batch_size = 64 
    transform = t.Compose([
        t.ToImage(), t.ToDtype(torch.float32), t.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#def load_model(model, weights):

if __name__ == "__main__":
    config = Config()
    wandb.init(project="ilsvrc", name = config.name, config = vars(config))

    dataset = ImagenetValidationDataset(config.transform)
    dataloader = DataLoader(dataset, shuffle = False, batch_size=config.batch_size, num_workers=config.num_workers)

    metric_params = {"task": "multiclass", "num_classes": 1000}
    metrics = torchmetrics.MetricCollection({
        "top1_accuracy": torchmetrics.Accuracy(**metric_params),
        "top5_accuracy": torchmetrics.Accuracy(**(metric_params | {"top_k": 5})),
        "precision": torchmetrics.Precision(**metric_params),
        "recall": torchmetrics.Recall(**metric_params),
        "f1": torchmetrics.F1Score(**metric_params),
    })

    step_metrics = metrics.clone(prefix="val/", postfix="_step")
    epoch_metrics = metrics.clone(prefix="val/", postfix="_epoch")

    model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights)
    model = model.to(config.device)
    model.eval()

    for step_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch: tuple[torch.Tensor, torch.Tensor]
        images, labels = batch
        images, labels = images.to(config.device), labels.to(config.device)
        preds = model(images, labels)

        preds, labels = preds.cpu(), labels.cpu()
        epoch_metrics.update(preds, labels)
        wandb.log(step_metrics(preds, labels) | {"step": step_idx})
        step_metrics.reset()

    wandb.log(epoch_metrics.compute(preds, labels))
    epoch_metrics.reset()

    wandb.finish()