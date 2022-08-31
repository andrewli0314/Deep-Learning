import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from config import device


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):    

# Data transformations

# Grayscale transformation was implemented to reduce from three channel images to a single channel.
# RandomAffine, RandomResizedCrop, RandomPerspective and RandomHorzontalFlip to able to determine the simpsons regardless of the angle of orientation. 
# RandomAutoContrast and RandomAdjustSharpness to ensure the lighting/quality of the images ensuring network's accuracy. 
# ToTensor converts the images into torch tensors and scales pixel values to 0.0 and 1.0 reducing the issue of exploding gradients. 

    if mode == 'train':
        return transforms.Compose(
            [
            transforms.Grayscale(), 
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.0, 0.5)),
            transforms.RandomResizedCrop((64, 64), scale=(0.5, 1.0)),
            transforms.RandomPerspective(p=0.2),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()
            ]
        )
    elif mode == 'test':
        return transforms.Compose(
            [
            transforms.Grayscale(),
            transforms.ToTensor(),
            ]
        )

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

  #  Architecture

  #  Conv2d, MaxPool, Batch Norm and Relu Layers
  #  Three Connected Layers with Relu, BatchNorm, Dropout was used on Second and Third Layer
  #  Using 3 by 3 to avoid small filters and small strides to capture features in small images.

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((5,5))
        self.linear_layers = nn.Sequential(
    
            nn.Linear(192*5*5, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 14),
            nn.BatchNorm1d(14)
        )

    def forward(self, t):
        f = self.cnn_layers(t)
        f = self.avgpool(f)
        f = torch.flatten(f, start_dim=1)
        f = self.linear_layers(f)
        return f


net = Network()
net.to(device)
lossFunc = nn.CrossEntropyLoss()


############################################################################
#######              Metaparameters and training options              ######
############################################################################

# Remain Default 

dataset = "./data"
train_val_split = 1
batch_size = 64
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.0006)
