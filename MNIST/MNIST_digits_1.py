import torchvision
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda:0')

train_dataset = torchvision.datasets.MNIST('/content', transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(784, 250)
        self.hidden = nn.Linear(250, 100)
        self.fc_out = nn.Linear(100, 10)

    def forward(self, x):
        # [32, 1, 28, 28]
        x = x.view(-1, 784)
        nn.Dropout()
        x = F.relu(self.fc_in(x))
        nn.Dropout()
        x = F.relu(self.hidden(x))
        x = self.fc_out(x)
        return x


model = Net()
model.to(device=device)
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    num_correct = 0
    num_examples = 0
    model.train()
    for batch in train_dataloader:
        input, target = batch
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], target)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
    print(f'Epoch: {epoch}, Loss: {loss:.2f}, Accuracy: {num_correct/num_examples:.2f}%')
