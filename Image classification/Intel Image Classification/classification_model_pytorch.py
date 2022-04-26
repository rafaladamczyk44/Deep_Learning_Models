import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

train_data_path = 'data/seg_train/seg_train'
test_data_path = 'data/seg_test/seg_test'

transforms = transforms.Compose([
	transforms.Resize((150, 150)),
	transforms.ToTensor()
])

# load data
train_dataset = ImageFolder(root=train_data_path, transform=transforms)
test_dataset = ImageFolder(root=test_data_path, transform=transforms)

batch_size = 32
val_size = 2000
train_size = len(train_dataset) - val_size

train_data, val_data = random_split(train_dataset, [train_size, val_size])


# num_workers – how many subprocesses to use for data loading.
# ``0`` means that the data will be loaded in the main process. (default: ``0``)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

device = torch.device('cuda:0')


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


def acc_score(output, labels):
	_, preds = torch.max(output, dim=1)
	sum_preds = torch.sum(preds == labels).item()
	size_of_preds = len(preds)
	result = torch.tensor(sum_preds / size_of_preds)
	return result
	

model = ImgClassifierNet()
model.to(device)

learning_rate = 2e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()
epochs = 20


def train(model, train_dataloader, optimizer):
	for epoch in range(epochs):
		model.train()
		
		for batch in train_dataloader:
			input, target = batch
			
			input = input.to(device)
			target = target.to(device)
		
			predict = model(input)
			loss = loss_fn(predict, target)
			
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
		print(f'Epoch: {epoch+1}, Loss: {loss:.2f}')


train(model=model, train_dataloader=train_dataloader, optimizer=optimizer)
torch.save(model.state_dict(), 'ImageClassifierState.pt')


