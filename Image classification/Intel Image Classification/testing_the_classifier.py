import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImgClassifierNet(nn.Module):
	def __init__(self):
		super(ImgClassifierNet, self).__init__()
		self.conv_layer1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
		)
		
		self.conv_layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
		)
		
		self.conv_layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2)
		)
		
		self.deep_layer = nn.Sequential(
			nn.Flatten(),
			nn.Linear(82944, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 6)
		)
	
	def forward(self, x):
		x = self.conv_layer1(x)
		x = self.conv_layer2(x)
		x = self.conv_layer3(x)
		output = self.deep_layer(x)
		return output


model = ImgClassifierNet()

model.load_state_dict(torch.load('ImageClassifierState.pt'))
model.eval()

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
