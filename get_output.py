"""Script to get output from the network"""

import torch

import model

# Load model
network = torch.nn.DataParallel(
        model.UNet(
            input_filters=1,
            unet_filters=64,
            N=2
        ))
network.load_state_dict(torch.load("ucsd_UNet.pth"))

# Load input image
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

image = np.array(Image.open('input.jpg'), dtype=np.float32) / 255
plt.imshow(image)
plt.show()

# Get output

# out = network(???)
