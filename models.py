import torch
import torch.nn.functional as F
import torch.nn as nn

class Representation(nn.Module):
    def __init__(self):
        super(Representation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # CIFAR-10 has 3 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 64, 256)  # Update to match new dimensions
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> [batch, 32, 32, 32]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> [batch, 32, 16, 16]
        x = F.relu(self.conv2(x))  # -> [batch, 64, 16, 16]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # -> [batch, 64, 8, 8]
        x = x.view(-1, 8 * 8 * 64)  # -> [batch, 4096]
        x = F.relu(self.fc1(x))  # -> [batch, 256]
        x = F.relu(self.fc2(x))  # -> [batch, 128]
        return x  # -> [batch, 128]

class Ten_class_classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.CNN = model
        self.mlp = nn.Linear(128, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.CNN(x)
        x = self.mlp(x)
        return x
    
class Four_class_classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.CNN = model
        self.mlp = nn.Linear(100, 4)

    def forward(self, x):
        x = self.CNN(x)
        x = self.mlp(x)
        return x
    
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()