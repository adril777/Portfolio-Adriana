import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models

# Load AlexNet pre-training
AlexNet = models.alexnet(pretrained=True)
AlexNet.eval()

# Obtain the weights of the second convolutional layer.
conv2_weights = AlexNet.features[3].weight.data

# Select the corresponding weights
weights_to_display = conv2_weights[:, 60, :, :][:32]

# Select the weights of the convolutional layer.
# In this case, we select the weights of the 61st filter (index 60 in zero-based indexing).
# [:32] limits the selection to the first 32 filters for visualization.

# Create the figure to show the weights
fig, ax = plt.subplots(nrows=4, ncols=8)
fig.set_size_inches(15, 7)

# Setting figure size to 15 inches wide and 7 inches high.

# Display weights in individual subplots
for i in range(32):
    ax[np.unravel_index(i, (4, 8))].imshow(weights_to_display[i].detach().numpy(), 'gray')
    ax[np.unravel_index(i, (4, 8))].set_title(f'Filter {i+1}')
    ax[np.unravel_index(i, (4, 8))].axis('off')


plt.tight_layout()
plt.show()
