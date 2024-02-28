
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#Dataset class
from torchvision import transforms

class ImageGPSDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None):
        self.image_paths = image_paths
        self.coordinates = coordinates
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),  # Resize to a uniform size
                transforms.ToTensor()           # Convert PIL Image to PyTorch Tensor
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        coords = self.coordinates[idx]
        coords = torch.tensor(coords, dtype=torch.float32)
        return image, coords

#Simplfied verison just using resnet 
class ResnetGPSModel(nn.Module):
    def __init__(self):
        super(ResnetGPSModel, self).__init__() 
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        '''I may need to freeze early layers
            for param in self.base_model.parameters():
                param.requires_grad = False'''
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2) 
        )

    def forward(self, x):
        return self.base_model(x)  # (lat,lng)

#I may need to expand this but here is the start 
class ImageGPSModelV1(nn.Module):
    def __init__(self):
        super(ImageGPSModelV1, self).__init__()
        # Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)  

    def forward(self, x):
        #conv + batch norm + activation + pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) #(lat,lng)
        return x

class ImageGPSModelV2(nn.Module):
    def __init__(self):
        super(ImageGPSModelV2, self).__init__()
        # Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(200704, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)  

    def forward(self, x):
        #conv + batch norm + activation + pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) #(lat,lng)
        return x
    
class ImageGPSModelV3(nn.Module):
    def __init__(self):
        super(ImageGPSModelV3, self).__init__()
        #Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted Fully Connected Layers
    
        self.fc1 = nn.Linear(65536, 512)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 2) 

    def forward(self, x):
        # Reduced layers in forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        x = self.fc3(x) #(lat,lng)
        return x

#first does classification then 
class ImageCityModel(nn.Module):
    def __init__(self):
        super(ImageCityModel, self).__init__()
        #Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusted Fully Connected Layers
    
        self.fc1 = nn.Linear(65536, 512)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)  
        self.fc3 = nn.Linear(256, 3)  #  for 3 cities

    def forward(self, x):
        # Reduced layers in forward pass
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        x = self.fc3(x) #(lat,lng)
        return F.log_softmax(x, dim=1) #for classification

