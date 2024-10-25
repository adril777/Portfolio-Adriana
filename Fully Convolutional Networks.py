import os, random
import torch
import torchvision
import numpy as np
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

# Initialization of the ResNet-18 model pre-trained on ImageNet:
pretrained_net = torchvision.models.resnet18(pretrained=True)

# Print all layers of the pre-trained ResNet-18 network
print(list(pretrained_net.children()))

# Create a new Sequential network (fully convolutional) by duplicating all layers except the last two from the member variable features of pretrained_net
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# Disable gradient calculation for all parameters in net so the weights of the pre-trained layers do not update.
for param in net.parameters():
    param.requires_grad = False

# Print the new Sequential network without the last two layers (global averaging layer and fully connected layer)
print(net)

# Define the input tensor X with dimensions (batch_size=1, channels=3, height=256, width=256)
X = torch.rand(size=(1, 3, 256, 256))

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net.to(device
# Mover el tensor de entra X a CUDA (si está disponible)
X = X.to(device)

# Compute the output of net for the input tensor X
output = net(X)

# Move the output back to the CPU for visualization (shape)
output_shape = output.cpu().shape
print(output_shape)

num_classes = 21  

# Add a 1x1 convolutional layer to adjust the number of channels to num_classes
net.add_module('final_conv', nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1))

# Define a function to generate a random kernel
def random_kernel(in_channels, out_channels, kernel_size):
    return torch.randn(out_channels, in_channels, kernel_size, kernel_size)

# Define the transposed convolutional layer with random initialization
conv_trans = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=64, padding=16, stride=32, bias=False)
conv_trans.weight.data.copy_(random_kernel(num_classes, num_classes, 64))

# Print the random kernel to verify
print('Random kernel (64 by 64):\n', conv_trans.weight.data[1, 1, :, :])

num_classes = 21

# Initialization of the random kernel for the transposed convolution layer
W = random_kernel(num_classes, num_classes, 64)

# Xavier initialization for the 1x1 convolutional layer
nn.init.xavier_uniform_(net.final_conv.weight)

# Copy the weights from the random kernel to the transposed convolutional layer in network 'net'
net.add_module('transpose_conv', nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=64, padding=16, stride=32))
net.transpose_conv.weight.data.copy_(W)

# Attempt to use all available GPUs if they are available
devices = d2l.try_all_gpus()

def predict(img):
   # Normalize the input image using the `normalize_image` function from the test (or validation) dataset
    X = img.unsqueeze(0).to(torch.float32)
    
   # Perform prediction using the model `net` on device `devices[0]` (likely GPU)
    pred = net(X.to(devices[0])).argmax(dim=1)
    
    # Reshape the prediction to match the dimensions of the original image
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
   # Retrieve the VOC_COLORMAP color map and move it to device
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    
    X = pred.long()
 
    
    return colormap[X, :]

#---------------------------------------------------------------------------------------

# Function to read images from the VOC2012 dataset
def read_voc_images(voc_dir, n, is_train=True):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    random.shuffle(images)
    for i in range(n):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{images[i]}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{images[i]}.png'),
                mode))
    return features, labels

# Directory of the VOC2012 dataset
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
n = 10  

# Read test images from the VOC2012 dataset
test_images, test_labels = read_voc_images(voc_dir, n, is_train=False)

imgs = []

# Iterate over each image to make predictions
for i in range(n):
   # Define the crop area (320x480 from the top-left corner)
    crop_rect = (0, 0, 320, 480)
    
   # Crop the area from the input image
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    
   # Perform prediction on the cropped area
    pred = label2image(predict(X))
    
    imgs += [
        X.permute(1, 2, 0),
        pred.cpu(),
        torchvision.transforms.functional.crop(test_labels[i],
                                               *crop_rect).permute(1, 2, 0)]

d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)

plt.tight_layout()
plt.show()
