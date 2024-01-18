#%%
import torch
import torch.nn as nn
from torchdiffeq import odeint
#%%
class Convblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Convblock,self).__init__()
        self.out_c = out_channel
        self.in_c = in_channel
        self.conv2d = nn.Conv2d(self.in_c,self.out_c,3,stride=1,padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


#%%
class ContractingPath(nn.Module):
    def __init__(self, input_window):
        super(ContractingPath,self).__init__()
        self.convb1_1 = Convblock(input_window,32) #여기 앞에 숫ㅏ
        self.convb1_2 = Convblock(32,32)
        self.convb2_1 = Convblock(32,64)
        self.convb2_2 = Convblock(64,64)
        self.convb3_1 = Convblock(64,128)
        self.convb3_2 = Convblock(128,128)
        self.maxpool = nn.MaxPool2d(2)


    def forward(self,x):
        x1 = self.convb1_1(x)
        x1 = self.convb1_2(x1)
        p1 = self.maxpool(x1)
        x2 = self.convb2_1(p1)
        x2 = self.convb2_2(x2)
        p2 = self.maxpool(x2)
        x3 = self.convb3_1(p2)
        x3 = self.convb3_2(x3)
        p3 = self.maxpool(x3)

        return x1,x2,x3,p3
    
#%%
class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck,self).__init__()
        self.convb1 = Convblock(128,256)
        self.convb2 = Convblock(256,256)


    def forward(self,x):
        
        x = self.convb1(x)
        x = self.convb2(x)

        return x
#%%
class ExpandingPath(nn.Module):
    def __init__(self, output_window):
        super(ExpandingPath,self).__init__()
        self.convb1 = Convblock(256,128)
        self.convb2 = Convblock(128,64)
        self.convb3 = Convblock(64,32)
        self.convb1_2 = Convblock(128,128)
        self.convb2_2 = Convblock(64,64)
        self.convb3_2 = Convblock(32,32)
        self.upconv1 = nn.ConvTranspose2d(256,128,4,stride=2,padding=1)
        self.upconv2 = nn.ConvTranspose2d(128,64,4,stride=2,padding=1)
        self.upconv3 = nn.ConvTranspose2d(64,32,4,stride=2,padding=1)
        self.upconv4 = nn.ConvTranspose2d(32,output_window,3,stride=1,padding=1)
        #self.sigmoid = nn.Sigmoid()
        

    def forward(self,x1,x2,x3,d):
        d = self.upconv1(d)
        d1 = torch.concat([d, x3], dim =1)
        d1 = self.convb1(d1)
        d1 = self.convb1_2(d1)
        d2 = self.upconv2(d1)
        d2 = torch.concat([d2, x2], dim =1)
        d2 = self.convb2(d2)
        d2 = self.convb2_2(d2)
        d3 = self.upconv3(d2)
        d3 = torch.concat([d3, x1], dim =1)
        d3 = self.convb3(d3)
        d3 = self.convb3_2(d3)
        out = self.upconv4(d3)
        #out = self.sigmoid(out)


        return out
#%%
class Unet(nn.Module):
    def __init__(self, input_window, output_window):
        super(Unet,self).__init__()
        self.contract = ContractingPath(input_window)
        self.bottleneck = BottleNeck()
        self.expand = ExpandingPath(output_window)

    def forward(self,x):
        x1, x2, x3, p3 = self.contract(x)
        p4 = self.bottleneck(p3)
        out = self.expand(x1,x2,x3,p4)

        return out
    
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
class ODEBlock(nn.Module):
    def __init__(self, ode_func):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func

    def forward(self, x, t):
        return odeint(self.ode_func, x, t, method='rk4')
#%%
class LinODEFunc(nn.Module):
    def __init__(self, input_window):
        super(LinODEFunc, self).__init__()
        self.lin = nn.Linear(input_window,input_window)

    def forward(self, t, x):
        x = x.permute(0,2,3,1)
        x = self.lin(x)
        x = x.permute(0,3,1,2)
        return x
#%%
class UnetNODE(nn.Module):
    def __init__(self,input_window, output_window,de,pe):
        super(UnetNODE, self).__init__()
        self.DCMP = DCMP_block(de, pe) # moving average
        self.ode_trend = ODEBlock(LinODEFunc(input_window))  # U-Net for trend
        self.ode_seasonal = ODEBlock(LinODEFunc(input_window))  # Convolution for seasonal
        self.ode_residual = ODEBlock(LinODEFunc(input_window))  # Convolution for residual
        #self.conv_trend = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1, stride=1, padding=0)
        #self.conv_seasonal = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1, stride=1, padding=0)
        #self.conv_residual = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.unet_trend = Unet(input_window, output_window)
        self.unet_seasonal = Unet(input_window, output_window)
        self.unet_residual = Unet(input_window, output_window)
        self.sigmoid = nn.Sigmoid()
        self.t = torch.linspace(0,1,7)
        
    def forward(self, x):
        t = self.t
        
        T, S, R = self.DCMP(x)
        T = self.ode_trend(T, t)[-1]
        S = self.ode_seasonal(S, t)[-1]
        R = self.ode_residual(R, t)[-1]
        
        T = self.unet_trend(T)
        S = self.unet_seasonal(S)
        R = self.unet_residual(R)
        
        x = T + S + R
        x = self.sigmoid(x)
        return x

