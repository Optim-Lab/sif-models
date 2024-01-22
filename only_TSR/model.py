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
class Seasonality(nn.Module):
    def __init__(self, period):
        super(Seasonality,self).__init__()
        self.period = period
        
    def forward(self, x):
        B, C, H, W = x.shape
        S = torch.zeros_like(x)

        for i in range(self.period):
            indices = torch.arange(i, C, self.period)
            seasonal_avg = torch.mean(x[:, indices, :, :], dim=1, keepdim=True)
            S[:, indices, :, :] = seasonal_avg.repeat(1, len(indices), 1, 1)
        
        return S
#%%
class DCMP_block(nn.Module):
    def __init__(self, kernel_size, pe):
        super(DCMP_block, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.season = Seasonality(period = pe)
        
    def forward(self, x):
        T = self.moving_avg(x)
        x = x - T
        
        S = self.season(x)
        R = x - S
        
        return T, S, R
#%%
class TSR(nn.Module):
    def __init__(self, input_window, output_window, de, pe):
        super(TSR, self).__init__()
        self.DCMP = DCMP_block(de,pe)
        self.lin_T = nn.Linear(input_window,output_window)
        self.lin_S = nn.Linear(input_window,output_window)
        self.lin_R = nn.Linear(input_window,output_window)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        T,S,R = self.DCMP(x)
        
        T = T.permute(0,2,3,1)
        T = self.lin_T(T)
        T = T.permute(0,3,1,2)
        
        S = S.permute(0,2,3,1)
        S = self.lin_S(S)
        S = S.permute(0,3,1,2)
        
        R = R.permute(0,2,3,1)
        R = self.lin_R(R)
        R = R.permute(0,3,1,2)
        
        x = T + S + R
        x = self.sigmoid(x)
        return x