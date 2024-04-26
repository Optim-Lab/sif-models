#%%
import torch.nn as nn
import torch.nn.functional as F
#%%
class CustomCNN(nn.Module):
    def __init__(self, input_window, output_window):
        super(CustomCNN, self).__init__()
        
        self.output_window = output_window
        self.conv1 = nn.Conv2d(in_channels=input_window, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(8 * 112 * 76, 256)  
        
        self.fc2 = nn.Linear(256, 448*304*output_window)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8 * 112 * 76)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.view(-1, self.output_window, 448, 304)
        x = self.sigmoid(x)

        return x

