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
class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel // 8, in_channel, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        max_out = self.shared_mlp(max_pooled)
        avg_out = self.shared_mlp(avg_pooled)
        out = self.sigmoid(max_out + avg_out)  
        return x * out 

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.channel_attention(x)

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attention = self.spatial_attention(pool_cat)
        return spatial_attention * x  

class CNNCBAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNNCBAM, self).__init__()
        self.cnn = Convblock(in_channel, out_channel)
        self.cbam = CBAM(out_channel)

    def forward(self, x):
        return self.cbam(self.cnn(x))



class ResnetCBAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ResnetCBAM,self).__init__()
        self.cnn1 = Convblock(in_channel,out_channel)
        self.cnn2 = Convblock(out_channel,out_channel)
        self.cbam = CBAM(out_channel)

        self.fit_channel = nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride =1, padding = 0)
        self.relu = nn.ReLU()

    def forward(self,x):
        res = self.fit_channel(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cbam(x)

        x = res + x
        x = self.relu(x)

        return x

#%%
class ContractingPath(nn.Module):
    def __init__(self, input_window):
        super(ContractingPath,self).__init__()
        self.cnn_cbam = CNNCBAM(input_window,input_window * 2) 

        self.resnet_cbam1_1 = ResnetCBAM(input_window*2, input_window*4)
        self.resnet_cbam1_2 = ResnetCBAM(input_window*4, input_window*4)

        self.resnet_cbam2_1 = ResnetCBAM(input_window*4, input_window*6)
        self.resnet_cbam2_2 = ResnetCBAM(input_window*6, input_window*6)

        self.resnet_cbam3_1 = ResnetCBAM(input_window*6, input_window*8)
        self.resnet_cbam3_2 = ResnetCBAM(input_window*8, input_window*8)

        self.resnet_cbam4 = ResnetCBAM(input_window*8, input_window*10)

        self.maxpool = nn.MaxPool2d(2)


    def forward(self,x):
        x1 = self.cnn_cbam(x)
        p1 = self.maxpool(x1)

        x2 = self.resnet_cbam1_1(p1)
        x2 = self.resnet_cbam1_2(x2)
        p2 = self.maxpool(x2)
        
        x3 = self.resnet_cbam2_1(p2)
        x3 = self.resnet_cbam2_2(x3)
        p3 = self.maxpool(x3)
        
        x4 = self.resnet_cbam3_1(p3)
        x4 = self.resnet_cbam3_2(x4)
        p4 = self.maxpool(x4)

        x5 = self.resnet_cbam4(p4)

        return x1,x2,x3,x4,x5
    
#%%
class ExpandingPath(nn.Module):
    def __init__(self, input_window, output_window):
        super(ExpandingPath,self).__init__()
        self.resnet_cbam_1 = ResnetCBAM(input_window*10, input_window*8)

        self.resnet_cbam_2_1 = ResnetCBAM(input_window*16, input_window*8)
        self.resnet_cbam_2_2 = ResnetCBAM(input_window*8, input_window*6)

        self.resnet_cbam_3_1 = ResnetCBAM(input_window*12, input_window*6)
        self.resnet_cbam_3_2 = ResnetCBAM(input_window*6, input_window*4)

        self.resnet_cbam_4_1 = ResnetCBAM(input_window*8, input_window*4)
        self.resnet_cbam_4_2 = ResnetCBAM(input_window*4, input_window*2)

        self.resnet_cbam_5_1 = ResnetCBAM(input_window*4, input_window*2)
        self.resnet_cbam_5_2 = ResnetCBAM(input_window*2, input_window)

        self.upconv1 = nn.ConvTranspose2d(input_window*8,input_window*8,2,stride=2,padding=0)
        self.upconv2 = nn.ConvTranspose2d(input_window*6,input_window*6,2,stride=2,padding=0)
        self.upconv3 = nn.ConvTranspose2d(input_window*4,input_window*4,2,stride=2,padding=0)
        self.upconv4 = nn.ConvTranspose2d(input_window*2,input_window*2,2,stride=2,padding=0)
        self.last = nn.Conv2d(input_window,output_window,1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x1,x2,x3,x4,x5):
        x5 = self.resnet_cbam_1(x5)
        x5 = self.upconv1(x5)
        x4 = torch.concat([x4, x5], dim =1)

        x4 = self.resnet_cbam_2_1(x4)
        x4 = self.resnet_cbam_2_2(x4)
        x4 = self.upconv2(x4)
        x3 = torch.concat([x3, x4], dim =1)

        x3 = self.resnet_cbam_3_1(x3)
        x3 = self.resnet_cbam_3_2(x3)
        x3 = self.upconv3(x3)
        x2 = torch.concat([x2, x3], dim =1)

        x2 = self.resnet_cbam_4_1(x2)
        x2 = self.resnet_cbam_4_2(x2)
        x2 = self.upconv4(x2)
        x1 = torch.concat([x1, x2], dim =1)

        x1 = self.resnet_cbam_5_1(x1)
        x1 = self.resnet_cbam_5_2(x1)
        x1 = self.last(x1)
        x1 = self.sigmoid(x1)

        return x1
#%%
class SICNet(nn.Module):
    def __init__(self, input_window, output_window):
        super(SICNet,self).__init__()
        self.contract = ContractingPath(input_window)
        self.expand = ExpandingPath(input_window, output_window)

    def forward(self,x):
        x1, x2, x3, x4, x5 = self.contract(x)
        out = self.expand(x1,x2,x3,x4,x5)

        return out