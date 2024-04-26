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


class ContractingPathSkip(nn.Module):
    def __init__(self, input_window):
        super(ContractingPathSkip,self).__init__()
        self.convb1_1 = Convblock(input_window,16)
        self.convb1_2 = Convblock(16,16)
        self.convb2_1 = Convblock(16,32)
        self.convb2_2 = Convblock(32,32)
        self.convb3_1 = Convblock(32,64)
        self.convb3_2 = Convblock(64,64)
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

        return x1,x2,x3

class ContractingPathValue(nn.Module):
    def __init__(self, input_window):
        super(ContractingPathValue,self).__init__()
        self.convb1_1 = Convblock(input_window,32)
        self.convb1_2 = Convblock(32,32)
        self.convb2_1 = Convblock(32,64)
        self.convb2_2 = Convblock(64,64)
        self.convb3_1 = Convblock(64,128)
        self.convb3_2 = Convblock(128,128)
        self.maxpool = nn.MaxPool2d(2)

        self.imagefusion1 = nn.Conv2d(32 + 16, 32, 1, 1)
        self.imagefusion2 = nn.Conv2d(64 + 32, 64, 1, 1)
        self.imagefusion3 = nn.Conv2d(128 + 64, 128, 1, 1)


    def forward(self,x, sub1,sub2,sub3):
        x1 = self.convb1_1(x)
        x1 = self.convb1_2(x1)
        x1 = torch.concat([x1, sub1], dim=1)
        x1 = self.imagefusion1(x1)
        p1 = self.maxpool(x1)
        
        x2 = self.convb2_1(p1)
        x2 = self.convb2_2(x2)
        x2 = torch.concat([x2, sub2], dim=1)
        x2 = self.imagefusion2(x2)
        p2 = self.maxpool(x2)
        
        x3 = self.convb3_1(p2)
        x3 = self.convb3_2(x3)
        x3 = torch.concat([x3, sub3], dim=1)
        x3 = self.imagefusion3(x3)
        p3 = self.maxpool(x3)

        return x1, x2, x3, p3

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
        self.last = nn.ConvTranspose2d(32,1,3,stride=1,padding=1)
        

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
        out = self.last(d3)


        return out

#%%
class ODEBlock(nn.Module):
    def __init__(self, ode_func):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func

    def forward(self, x, t):

        return odeint(self.ode_func, x, t, method = 'rk4')

class ConvODEFunc(nn.Module):
    def __init__(self,channel):
        super(ConvODEFunc, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)

    def forward(self, t, x):
        
        return self.conv(x)


#%%
class Unet(nn.Module):
    def __init__(self, input_window, output_window):
        super(Unet,self).__init__()
        self.output_window = output_window
        self.contract_sub = ContractingPathSkip(3)
        self.contract_sic = ContractingPathValue(input_window)
        self.bottleneck = BottleNeck()
        self.expand = nn.ModuleList()
        for j in range(self.output_window):
            self.expand.append(ExpandingPath(output_window))
        self.ode = ODEBlock(ConvODEFunc(256))
        self.t = torch.linspace(0,3,4)
        

    def forward(self,x,sub):
        t = self.t

        
        x1, x2, x3 = self.contract_sub(sub)

        x1, x2, x3, p3 = self.contract_sic(x, x1, x2, x3)
        p3 = self.bottleneck(p3)
        p3 = self.ode(p3, t)

        out = torch.zeros([x.size(0),self.output_window,x.size(2),x.size(3)],dtype=x.dtype).to(x.device)

        for i in range(self.output_window):
            out[:,i,:,:] = torch.squeeze(self.expand[i](x1,x2,x3,p3[i]))

        return out        

#%%
class UnetNODECombined(nn.Module):
    def __init__(self, input_window,output_window):
        super(UnetNODECombined,self).__init__()
        self.unet = Unet(input_window,output_window)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x1,x2,x3,x4):

        x2 = torch.cat([x2[:,-1,:,:].unsqueeze(1),x3[:,-1,:,:].unsqueeze(1),x4[:,-1,:,:].unsqueeze(1)],dim=1)

        out = self.unet(x1,x2)

        return self.sigmoid(out)

