#%%
import torch
import torch.nn as nn
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
        self.convb1_1 = Convblock(input_window,32)
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
        self.upconv1 = nn.ConvTranspose2d(256,128,2,stride=2,padding=0)
        self.upconv2 = nn.ConvTranspose2d(128,64,2,stride=2,padding=0)
        self.upconv3 = nn.ConvTranspose2d(64,32,2,stride=2,padding=0)
        self.upconv4 = nn.ConvTranspose2d(32,output_window,3,stride=1,padding=1)
        

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
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
#%%
class UnetDLinear(nn.Module):
    def __init__(self, input_window, output_window, de):
        super(UnetDLinear,self).__init__()
        self.decomp = series_decomp(de)
        self.unetT = Unet(input_window, output_window)
        self.unetR = Unet(input_window, output_window)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        res, moving_mean = self.decomp(x)
        res = self.unetR(res)
        moving_mean = self.unetT(moving_mean)
        x = res + moving_mean
        x = self.sigmoid(x)
        
        return x