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

image =Image.open('input.jpg')
npimg = np.array(image, dtype=np.float32) / 255
#plt.imshow(image)
#plt.show()

# Get output
from torchvision.transforms import ToTensor
tensorimg = ToTensor()(image)
out = network(tensorimg)

print(out)
