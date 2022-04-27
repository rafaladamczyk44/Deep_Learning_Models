import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from classification_model_pytorch import train

device = torch.device('cuda:0')

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

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

transfer_model = torchvision.models.resnet50(pretrained=True)
# Freeze parameters other than batch norm
for name, param in transfer_model.named_parameters():
	if 'bn' not in name:
		param.requires_grad = False


learning_rate = 2e-2
optimizer = optim.AdamW(transfer_model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()
epochs = 20

train(model=transfer_model, train_dataloader=train_dataloader, optimizer=optimizer)
