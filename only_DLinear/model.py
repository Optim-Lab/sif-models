#%%
import torch
import torch.nn as nn
#%%
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B,C,H*W)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.reshape(B,C,H,W)
        return x
    
#%%
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
#%%
class DLinear(nn.Module):
    def __init__(self, input_window, output_window, de):
        super(DLinear, self).__init__() 
        self.DCMP = series_decomp(de)
        self.lin_T = nn.Linear(input_window,output_window)
        self.lin_R = nn.Linear(input_window,output_window)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        T,R = self.DCMP(x)
        
        T = T.permute(0,2,3,1)
        T = self.lin_T(T)
        T = T.permute(0,3,1,2)

        R = R.permute(0,2,3,1)
        R = self.lin_R(R)
        R = R.permute(0,3,1,2)
        
        x = T + R
        x = self.sigmoid(x)
        return x