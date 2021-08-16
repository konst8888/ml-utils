import torch.nn as nn

class AudioCNN(nn.Module):
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 4)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d((2, 4))
        self.conv2 = nn.Conv2d(64, 64, (3, 5))
        self.mp2 = nn.MaxPool2d(2)
        self.drop2d = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(7296, 64)
        self.drop = nn.Dropout(0.2)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.drop2d(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = self.drop2d(x)
        
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.lin3(x)
        
        return x