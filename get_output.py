"""Script to get output from the network"""

import torch

import model

network = torch.nn.DataParallel(
        model.UNet(
            input_filters=1,
            unet_filters=64,
            N=2
        ))
network.load_state_dict(torch.load("ucsd_UNet.pth"))

