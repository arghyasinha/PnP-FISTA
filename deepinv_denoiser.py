# import torchvision.transforms as transforms
import torch
import numpy as np
from deepinv.models import DnCNN
import deepinv as dinv
from deepinv.utils.plotting import plot, plot_curves



n_channels = 1  # number of color channels in the input image

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"



denoiser_dncnn = DnCNN(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="./models/dncnn_sigma2_lipschitz_gray.pth",  # automatically downloads the pretrained weights, set to a path to use custom weights.
    # pretrained="download_lipchitz",
    device=device,
)
denoiser_drunet = dinv.models.DRUNet(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="./models/drunet_gray.pth",
    # pretrained="download",
    device=device,
)
#denoiser_TV = dinv.models.TVDenoiser()


def run_deepinv(y,sigma = 0.05,denoiser_deep = denoiser_dncnn,plot_curves=False):
    y_ = torch.from_numpy(y.astype(np.float32)).view(1, -1, y.shape[0], y.shape[1])
    y_ = y_.to(device)
    x_ = denoiser_deep(y_,sigma = sigma)  # inference
    if plot_curves:
        plot(x_)

    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float32)
    x_ = x_.reshape(y.shape)
    return x_
