"""Script to get output from the network"""

import torch

import model

print('# Load model')
network = torch.nn.DataParallel(
        model.UNet(
            input_filters=1,
            unet_filters=64,
            N=2
        ))
network.load_state_dict(torch.load("ucsd_UNet.pth"))

print('# Load input image')
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import sys

if sys.argv[1:]:
    image = Image.open(sys.argv[1])
else:
    sys.exit('usage: ' + sys.argv[0] + ' <input.png>')

#npimg = np.array(image, dtype=np.float32) / 255
img1ch = image.convert('L')
#plt.imshow(img1ch)
#plt.show()

print("# Get output")
from torchvision.transforms import ToTensor, ToPILImage
tensorimg = ToTensor()(img1ch).unsqueeze(0)
out = network(tensorimg)

print(out.shape)
print(out)

print("# Count results")
# density maps were normalized to 100 * no. of objects
# to make network learn better
#print(len(out), type(out))
print(torch.sum(out[0]).item() / 100)

print("# Plot output")
#plt.imshow(out.detach().numpy().squeeze(0))
plt.imshow(ToPILImage()(out[0]))
plt.show()
