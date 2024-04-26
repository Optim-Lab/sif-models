#%%
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
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



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size != 0:
            x = x[:, :, :-self.chomp_size].contiguous()
        else: x 
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TSAM(nn.Module):
    def __init__(self, channels):
        super(TSAM, self).__init__()
        self.channels = channels
        kernel_size_tcn = max(channels // 12, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_tcn1 = TCN(num_inputs = 1, num_channels = [8,8,8], kernel_size = channels//12)
        self.max_tcn2 = TCN(num_inputs = 8, num_channels = [8,8,8], kernel_size = channels//12)
        self.max_tcn3 = TCN(num_inputs = 8, num_channels = [1,1], kernel_size = 1)
        self.avg_tcn1 = TCN(num_inputs = 1, num_channels = [8,8,8], kernel_size = channels//12)
        self.avg_tcn2 = TCN(num_inputs = 8, num_channels = [8,8,8], kernel_size = channels//12)
        self.avg_tcn3 = TCN(num_inputs = 8, num_channels = [1,1], kernel_size= 1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pooled = self.max_pool(x)

        avg_pooled = self.avg_pool(x)

        max_out = self.max_tcn3(self.max_tcn2(self.max_tcn1(torch.squeeze(max_pooled,-1).permute(0,2,1)))).permute(0,2,1)
        avg_out = self.avg_tcn3(self.avg_tcn2(self.avg_tcn1(torch.squeeze(avg_pooled,-1).permute(0,2,1)))).permute(0,2,1)

        tcn_out = max_out + avg_out
        tcn_attention = self.sigmoid(tcn_out.unsqueeze(-1))
        x = tcn_attention * x

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_cat = torch.cat([max_pool, avg_pool], dim=1)
        spatial_attention = self.spatial_conv(pool_cat)

        x = x * self.sigmoid(spatial_attention)

        return x


class CNNTSAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNNTSAM, self).__init__()
        self.cnn = Convblock(in_channel, out_channel)
        self.tsam = TSAM(out_channel)

    def forward(self, x):
        return self.tsam(self.cnn(x))



class ResnetTSAM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ResnetTSAM,self).__init__()
        self.cnn1 = Convblock(in_channel,out_channel)
        self.cnn2 = Convblock(out_channel,out_channel)
        self.tsam = TSAM(out_channel)

        self.fit_channel = nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride =1, padding = 0)
        self.relu = nn.ReLU()

    def forward(self,x):
        res = self.fit_channel(x)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.tsam(x)

        x = res + x
        x = self.relu(x)

        return x

#%%
class ContractingPath(nn.Module):
    def __init__(self, input_window):
        super(ContractingPath,self).__init__()
        self.cnn_tsam = CNNTSAM(input_window,input_window * 2) 

        self.resnet_tsam1_1 = ResnetTSAM(input_window*2, input_window*4)
        self.resnet_tsam1_2 = ResnetTSAM(input_window*4, input_window*4)

        self.resnet_tsam2_1 = ResnetTSAM(input_window*4, input_window*6)
        self.resnet_tsam2_2 = ResnetTSAM(input_window*6, input_window*6)

        self.resnet_tsam3_1 = ResnetTSAM(input_window*6, input_window*8)
        self.resnet_tsam3_2 = ResnetTSAM(input_window*8, input_window*8)

        self.resnet_tsam4 = ResnetTSAM(input_window*8, input_window*10)

        self.maxpool = nn.MaxPool2d(2)


    def forward(self,x):
        x1 = self.cnn_tsam(x)
        p1 = self.maxpool(x1)

        x2 = self.resnet_tsam1_1(p1)
        x2 = self.resnet_tsam1_2(x2)
        p2 = self.maxpool(x2)
        
        x3 = self.resnet_tsam2_1(p2)
        x3 = self.resnet_tsam2_2(x3)
        p3 = self.maxpool(x3)
        
        x4 = self.resnet_tsam3_1(p3)
        x4 = self.resnet_tsam3_2(x4)
        p4 = self.maxpool(x4)

        x5 = self.resnet_tsam4(p4)

        return x1,x2,x3,x4,x5
    
#%%
class ExpandingPath(nn.Module):
    def __init__(self, input_window, output_window):
        super(ExpandingPath,self).__init__()
        self.resnet_tsam_1 = ResnetTSAM(input_window*10, input_window*8)

        self.resnet_tsam_2_1 = ResnetTSAM(input_window*16, input_window*8)
        self.resnet_tsam_2_2 = ResnetTSAM(input_window*8, input_window*6)

        self.resnet_tsam_3_1 = ResnetTSAM(input_window*12, input_window*6)
        self.resnet_tsam_3_2 = ResnetTSAM(input_window*6, input_window*4)

        self.resnet_tsam_4_1 = ResnetTSAM(input_window*8, input_window*4)
        self.resnet_tsam_4_2 = ResnetTSAM(input_window*4, input_window*2)

        self.resnet_tsam_5_1 = ResnetTSAM(input_window*4, input_window*2)
        self.resnet_tsam_5_2 = ResnetTSAM(input_window*2, input_window)

        self.upconv1 = nn.ConvTranspose2d(input_window*8,input_window*8,2,stride=2,padding=0)
        self.upconv2 = nn.ConvTranspose2d(input_window*6,input_window*6,2,stride=2,padding=0)
        self.upconv3 = nn.ConvTranspose2d(input_window*4,input_window*4,2,stride=2,padding=0)
        self.upconv4 = nn.ConvTranspose2d(input_window*2,input_window*2,2,stride=2,padding=0)
        self.last = nn.Conv2d(input_window,output_window,1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x1,x2,x3,x4,x5):
        x5 = self.resnet_tsam_1(x5)
        x5 = self.upconv1(x5)
        x4 = torch.concat([x4, x5], dim =1)

        x4 = self.resnet_tsam_2_1(x4)
        x4 = self.resnet_tsam_2_2(x4)
        x4 = self.upconv2(x4)
        x3 = torch.concat([x3, x4], dim =1)

        x3 = self.resnet_tsam_3_1(x3)
        x3 = self.resnet_tsam_3_2(x3)
        x3 = self.upconv3(x3)
        x2 = torch.concat([x2, x3], dim =1)

        x2 = self.resnet_tsam_4_1(x2)
        x2 = self.resnet_tsam_4_2(x2)
        x2 = self.upconv4(x2)
        x1 = torch.concat([x1, x2], dim =1)

        x1 = self.resnet_tsam_5_1(x1)
        x1 = self.resnet_tsam_5_2(x1)
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