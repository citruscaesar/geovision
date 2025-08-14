from typing import Literal, Optional, Callable
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST 
# from string import digits, ascii_letters
from torchvision.transforms.v2 import Transform, Compose, ToImage, ToDtype, Normalize

from matplotlib.image import AxesImage
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 

class CFG:
    seed = 42
    log_every_n_steps = 50 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root: Path = Path.home() / "datasets"
    ckpts_home = Path.home() / "models" / "gans"
    class_names = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) 
    image_shape = (28, 28)
    noise_shape = (10, 10)
    dataloader_params: dict = {
        "batch_size": 100,
        "num_workers": 2,
        "prefetch_factor": 2, 
        "pin_memory": True,
    }
    image_transforms: Transform = Compose([ToImage(), ToDtype(torch.float32, True), Normalize((0.5,), (0.5,))]) 
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer: Callable[..., Optimizer] = torch.optim.RMSprop
    generator_optimizer_params = {
        "lr": 0.0005,
        #"betas" : (0.5, 0.999)
    }
    discriminator_optimizer_params = {
        "lr": 0.00005,
        #"betas" : (0.5, 0.999)
    }
    genenerator_step_interval = 2
    
class Metric:
    def __init__(self, length: int):
        assert isinstance(length, int)
        self.idx = 0
        self.buffer = np.full(length, fill_value=np.nan, dtype=np.float16)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, value: float):
        self.buffer[self.idx] = value
        self.idx += 1
    
    def compute(self) -> float:
        mean = np.nanmean(self.buffer)
        self.reset()
        return mean
    
    def reset(self) -> None:
        self.buffer[:] = np.nan
        self.idx = 0

    @property
    def last(self) -> float:
        return self.buffer[self.idx-1]

class Generator(torch.nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        # torch.nn.ConvTranspose2d(in_ch+1, 64, kernel_size=5, stride = 1, bias = False)) # 10x10 -> 14x14
        self.label_embedding = torch.nn.Embedding(len(CFG.class_names), np.prod(CFG.noise_shape))

        self.up_1 = torch.nn.Upsample(size = 14, mode = "bilinear")
        self.conv_1 = self._block(in_ch+1, 64)
        self.conv_2 = self._block(64, 64)

        self.up_2 = torch.nn.Upsample(size = 28, mode = "bilinear")
        self.conv_3 = self._block(64, 64)
        self.conv_4 = self._block(64, 64) 
       
        self.conv_5 = torch.nn.Sequential()
        self.conv_5.add_module("conv_1", torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias = False))
        self.conv_5.add_module("act_1", torch.nn.Tanh())
    
    def _block(self, in_ch: int, out_ch:int):
        layer = torch.nn.Sequential()
        layer.add_module("conv_1", torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias = False))
        layer.add_module("norm_1", torch.nn.BatchNorm2d(out_ch))
        layer.add_module("act_1", torch.nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, x: torch.Tensor, y: int):
        # print(x.shape, x.dtype, x.device)
        # print(y.shape, y.dtype, y.device)
        y = self.label_embedding(y).reshape((-1, 1, *CFG.noise_shape))#.reshape((1, *CFG.noise_shape))
        x = torch.concat([x, y], dim = 1)
        for layer in self.children():
            if layer == self.label_embedding:
                continue
            x = layer(x)
        return x 

class Discriminator(torch.nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.label_embedding = torch.nn.Embedding(len(CFG.class_names), np.prod(CFG.image_shape))

        self.conv_1 = self._block(in_ch+1, 16) 
        self.conv_2 = self._block(16, 32, downsample=True) 
        self.conv_3 = self._block(32, 64, downsample=True) 
        self.conv_4 = self._block(64, 64, downsample=True) 

        self.fc_5 = torch.nn.Sequential()
        self.fc_5.add_module("pool_1", torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.fc_5.add_module("flatten_1", torch.nn.Flatten(1))
        self.fc_5.add_module("fc_1", torch.nn.Linear(64, 1, bias = True))
        # self.fc_5.add_module("act_1", torch.nn.Sigmoid())
    
    def _block(self, in_ch: int, out_ch: int, downsample: bool = False):
        return torch.nn.Sequential(OrderedDict([
            ("conv_1", torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2 if downsample else 1, padding=1, bias = False)),
            ("norm_1", torch.nn.BatchNorm2d(out_ch)),
            ("act_1", torch.nn.LeakyReLU(0.2, inplace=True))
        ]))

    def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
        y = self.label_embedding(y).reshape((-1, 1, *CFG.image_shape))
        x = torch.concat([x, y], dim = 1)
        for layer in self.children():
            if layer == self.label_embedding:
                continue
            x = layer(x)
        return x

def get_dataset(split: Literal["train", "test"] = "train"):
    return MNIST(root = CFG.dataset_root, train = (split=="train"), download = True, transform = CFG.image_transforms)

def get_dataloader(split: Literal["train", "test"]):
    return DataLoader(get_dataset(split), shuffle = (split == "train"), **CFG.dataloader_params)

def get_model(which: Literal["generator", "discriminator"], split: Literal["train", "eval"], **kwargs) -> tuple[Module, Optimizer]:
    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)
            
    if which == "generator":
        optim_kwargs = CFG.generator_optimizer_params
        model_constructor = Generator
    else:
        optim_kwargs = CFG.discriminator_optimizer_params
        model_constructor = Discriminator
        
    model = model_constructor(**kwargs)
    model.apply(init_weights)
    model = model.train(split=="train")
    model = model.to(CFG.device)
    optimizier = CFG.optimizer(model.parameters(), **optim_kwargs)
    return model, optimizier 

def get_noise(batch_size: int = 1):
    return torch.randn((batch_size, 1, *CFG.noise_shape), dtype = torch.float32, device = CFG.device)

def get_metric_plots(steps: NDArray, gen_loss: Metric, disc_loss: Metric, ylim: float = 2.0):
    fig = plt.figure(figsize = (15, 5), layout = "constrained")
    gs = GridSpec(nrows = 1, ncols = 2, width_ratios=(7, 3), figure = fig)

    loss_ax = fig.add_subplot(gs[0, 0])
    loss_ax.set_xlim(0, len(steps))
    loss_ax.set_ylim(-1, ylim)
    gen_loss_plot, = loss_ax.plot(steps, gen_loss.buffer, animated = True, label = "generator")
    disc_loss_plot, = loss_ax.plot(steps, disc_loss.buffer, animated = True, label = "discriminator")
    loss_ax.legend(loc="upper left")

    n = CFG.dataloader_params["batch_size"] 
    nrows = int(np.around(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    gs = GridSpecFromSubplotSpec(nrows, ncols, gs[0, 1])
    image_axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]

    gen_plots: list[AxesImage] = list()
    for ax in image_axes:
        plot = ax.imshow(np.random.rand(28, 28), cmap = "gray", animated = True)
        ax.axis("off")
        gen_plots.append(plot)
    return fig, gen_loss_plot, disc_loss_plot, gen_plots 
 
def plot_grid(images: NDArray, labels: Optional[NDArray] = None):
    n = len(images)
    assert n >= 2, f"len(images) must be at least 2, got {n}"
    nrows = int(np.around(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    _, axes = plt.subplots(nrows, ncols, layout = "constrained", figsize = (6, 6))
    for idx, ax in enumerate(axes.ravel()):
        if idx < n:
            ax.imshow(images[idx].squeeze(), cmap = "grey")
            if labels is not None:
                ax.set_title(labels[idx])
        ax.axis("off")

def train_step(batch: tuple[Tensor, Tensor], batch_idx: int, gen: Module, disc: Module, gen_opt: Optimizer, disc_opt: Optimizer):
    images = batch[0].to(CFG.device)
    labels = batch[1].to(CFG.device)

    y_fake = torch.full((len(images), 1), fill_value = 0.0, dtype = torch.float32, device = CFG.device, requires_grad = False) 
    y_real = torch.full((len(images), 1), fill_value = 1.0, dtype = torch.float32, device = CFG.device, requires_grad = False) 

    # Discriminator Step
    disc_opt.zero_grad()
    z = torch.randn((len(images), 1, 10, 10), dtype = torch.float32, device = CFG.device)
    fake_loss = CFG.criterion(disc(gen(z, labels).detach(), labels), y_fake)
    real_loss = CFG.criterion(disc(images, labels), y_real)
    disc_loss = (fake_loss + real_loss) / 2
    disc_loss.backward()
    disc_opt.step()

    if (batch_idx + 1) % CFG.genenerator_step_interval == 0:
        # Generator Step
        gen_opt.zero_grad()
        z = torch.randn((len(images), 1, 10, 10), dtype = torch.float32, device = CFG.device)
        gen_loss = CFG.criterion(disc(gen(z, labels), labels), y_real)
        gen_loss.backward()
        gen_opt.step()
    else:
        gen_loss = torch.tensor(torch.nan) 
    return gen_loss, disc_loss 

def wgan_train_step(batch: tuple[Tensor, Tensor], batch_idx: int, gen: Module, disc: Module, gen_opt: Optimizer, disc_opt: Optimizer):

    def gradient_penalty(real_images: Tensor, gen_images: Tensor, lambda_gp: float = 10.0):
        alpha = torch.rand(len(real_images), 1, 1, 1, device=CFG.device)#.expand_as(real_images)
        interpolated = alpha * real_images + (1 - alpha) * gen_images 
        interpolated.requires_grad_(True)

        interpolated_scores = disc(interpolated, labels)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores, 
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores, device=CFG.device),
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]
        gradients = torch.linalg.vector_norm(gradients.view(len(real_images), -1), ord=2, dim=1)
        gradient_penalty = lambda_gp * ((gradients - 1) ** 2).mean()
        return gradient_penalty

    images = batch[0].to(CFG.device)
    labels = batch[1].to(CFG.device)

    # Discriminator Step
    disc_opt.zero_grad()
    z = torch.randn((len(images), 1, 10, 10), dtype = torch.float32, device = CFG.device)
    generated_images = gen(z, labels).detach()

    disc_loss = -(disc(images, labels).mean() - disc(generated_images, labels).mean()) + gradient_penalty(images, generated_images)
    disc_loss.backward()
    disc_opt.step()

    for param in disc.parameters():
        param.data.clamp_(-0.01, 0.1)

    if (batch_idx + 1) % CFG.genenerator_step_interval == 0:
        # Generator Step
        gen_opt.zero_grad()
        z = torch.randn((len(images), 1, 10, 10), dtype = torch.float32, device = CFG.device)
        gen_loss = -disc(gen(z, labels), labels).mean()
        gen_loss.backward()
        gen_opt.step()
    else:
        gen_loss = torch.tensor(torch.nan) 

    return gen_loss, disc_loss

if __name__ == "__main__":
    cli = argparse.ArgumentParser("standalone script to train and evaluate GANs on the EMNIST Handwriting Dataset")
    cli.add_argument("--mode", choices=["train", "eval"])
    cli.add_argument("--num_epochs", type = int, default = 1)
    cli.add_argument("--log_every_n_steps", type = int, default = CFG.log_every_n_steps)
    cli.add_argument("--load_epoch", type = int, default = None)
    cli.add_argument("--load_step", type = int, default = None)
    args = cli.parse_args()

    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    eval_noise = get_noise(CFG.dataloader_params["batch_size"])
    eval_labels = (torch.arange(0, 10)*torch.ones(10, 10)).flatten().to(torch.int64).to(CFG.device)
    
    if args.mode == "train":
        # TODO: update ylim on plots dynamically if loss grows beyond
        # if gen_losses.max() > loss_ax.get_ylim()[1]:
            # loss_ax.set_ylim(0, gen_losses.max() + 0.1)

        # init basic stuff 
        gen, gen_opt = get_model("generator", args.mode)
        disc, disc_opt = get_model("discriminator", args.mode)
        train_dl = get_dataloader("train")

        # init metric buffers
        num_logs = args.num_epochs * len(train_dl) // args.log_every_n_steps
        # print(f"num_epochs = {args.num_epochs} :: num_steps = {len(train_dl)} :: logging_interval = {args.log_every_n_steps}")
        steps = np.arange(num_logs, dtype = np.int32)
        gen_loss, disc_loss = Metric(num_logs), Metric(num_logs)
        gen_step_loss, disc_step_loss = Metric(args.log_every_n_steps), Metric(args.log_every_n_steps)

        # init plotting window
        fig, gen_loss_plot, disc_loss_plot, eval_plots = get_metric_plots(steps, gen_loss, disc_loss)
        def update(_):
            gen_loss_plot.set_ydata(gen_loss.buffer)
            disc_loss_plot.set_ydata(disc_loss.buffer)
            for plot, image in zip(eval_plots, gen(eval_noise, eval_labels).detach().squeeze().cpu().numpy()):
                plot.set_array(image)
            return gen_loss_plot, disc_loss_plot, *eval_plots
        loss_animation = FuncAnimation(fig = fig, func = update, blit = True, interval = 100, cache_frame_data=False)
        plt.ion()
        plt.show()

        # training loop
        last_step, last_epoch = 0, 0
        for epoch in range(args.num_epochs):
            for step, batch in enumerate(train_dl):
                _gen_loss, _disc_loss = wgan_train_step(batch, step, gen, disc, gen_opt, disc_opt)
                gen_step_loss.append(_gen_loss.item())
                disc_step_loss.append(_disc_loss.item())
                if (step+1) % args.log_every_n_steps == 0:
                    gen_loss.append(gen_step_loss.compute())
                    disc_loss.append(disc_step_loss.compute())
                    print(f"epoch = {epoch} :: step = {step} :: generator_loss = {gen_loss.last:.3f} :: discriminator_loss = {disc_loss.last:.3f}")
                    plt.pause(0.0001)
                last_step = step
            gen_step_loss.reset()
            disc_step_loss.reset()
            last_epoch = epoch

        # save ckpt
        CFG.ckpts_home.mkdir(exist_ok=True, parents=True)
        torch.save({"generator": gen.state_dict(), "discriminator": disc.state_dict()}, CFG.ckpts_home/f"gan_epoch={last_epoch}_step={last_step}.ckpt")
        plt.savefig("gan_training.png")

    if args.mode == "eval":
        assert isinstance(args.load_epoch, int), "missing checkpoint epoch"
        assert isinstance(args.load_step, int), "missing checkpoint step"
        
        with torch.no_grad():
            gen, _ = get_model("generator", args.mode)
            gen.load_state_dict(torch.load(CFG.ckpts_home/f"gan_epoch={last_epoch}_step={last_step}.ckpt", weights_only=True)["generator"])
            plot_grid(gen(eval_noise, eval_labels).squeeze().detach().cpu().numpy()) # [CFG.class_names[x.item()] for x in train_batch[1]]