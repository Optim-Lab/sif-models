#%%
import torch
import torch.nn as nn
#%%
class NLinear(nn.Module):
    def __init__(self, input_window, output_window):
        super(NLinear,self).__init__()
        self.linear = nn.Linear(input_window, output_window)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        seq_last = x[:,-1:,:,:].detach()
        x = x - seq_last

        x = x.permute(0,2,3,1)
        x = self.linear(x)
        x = x.permute(0,3,1,2)

        x = x + seq_last

        x = self.sigmoid(x)
        
        return x