import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2)
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2)
        self.conv10 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv12 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        
        self.fc1 = nn.Linear(256 * 9 * 9, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p = 0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm2d1 = nn.BatchNorm2d(32)
        self.batch_norm2d2 = nn.BatchNorm2d(128)
        self.batch_norm1d = nn.BatchNorm1d(128)
        
    
    def initialize_weights(self):
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=1.0, generator=None)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain=1.0, generator=None)
        torch.nn.init.xavier_normal_(self.fc3.weight, gain=1.0, generator=None)
        
    
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(F.relu(self.conv2(x1))) ### 150
        x3 = F.relu(self.conv3(x2))
        x4 = self.batch_norm2d1(self.pool(F.relu(self.conv4(x3)) + x3)) ### 75
        x5 = F.relu(self.conv5(x4))
        x6 = self.pool(F.relu(self.conv6(x5)) + x5) ### 37
        x7 = F.relu(self.conv7(x6))
        x8 = F.relu(self.conv8(x7))
        x9 = self.batch_norm2d2(self.pool(F.relu(self.conv9(x8)))) ### 18
        x10 = F.relu(self.conv10(x9))
        x11 = F.relu(self.conv11(x10))
        x12 = self.pool(F.relu(self.conv12(x11)) + x10) ### 9
        x13 = torch.flatten(x12, 1) # flatten all dimensions except batch
        x14 = F.relu(self.fc1(x13))
        x15 = self.dropout(x14)
        x16 = F.relu(self.batch_norm1d(self.fc2(x15)))
        x17 = self.fc3(x16)
        return x17
        