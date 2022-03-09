import torch
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
    
    
import torchvision.models as models

class AudioCNN(nn.Module):
    
    def __init__(self, num_classes=3):
        super().__init__()
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        mobilenet_v2.features[0][0] = nn.Conv2d(2, 32, kernel_size=3, stride=2, bias=False)
        mobilenet_v2.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.model = mobilenet_v2
        
    def forward(self, x):
        return self.model(x)
    
    
class AudioCNN(nn.Module):
    
    def __init__(self, num_classes=3):
        super().__init__()
        shufflenet_v2_x0_5 = models.shufflenet_v2_x0_5(pretrained=True)
        shufflenet_v2_x0_5.conv1[0] = nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        shufflenet_v2_x0_5.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.model = shufflenet_v2_x0_5
        
    def forward(self, x):
        return self.model(x)
    
    
class AudioCNN(nn.Module):
    
    def __init__(self, num_classes=3, n_features=0):
        super().__init__()
        shufflenet_v2_x0_5 = models.shufflenet_v2_x0_5(pretrained=True)
        shufflenet_v2_x0_5.conv1[0] = nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #shufflenet_v2_x0_5.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        shufflenet_v2_x0_5.fc = nn.Sequential()
        
        #self.main = nn.Sequential(*list(shufflenet_v2_x0_5.children())[:-1])
        self.main = shufflenet_v2_x0_5
        self.fc = nn.Linear(in_features=1024 + n_features, out_features=num_classes, bias=True)
        # maybe need to add one more fc
        self.n_features = n_features
        
    def forward(self, x):
        if self.n_features == 0:
            x = self.main(x)
            x = self.fc(x)
            return x
        
        imgs, features = x
        x = self.main(imgs)
        x = torch.cat([x, features], dim=1)
        x = self.fc(x)
        
        return x