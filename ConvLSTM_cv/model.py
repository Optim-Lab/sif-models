#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=self.padding, bias=bias)

    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        input_tensor = input_tensor.unsqueeze(1) if input_tensor.dim() == 3 else input_tensor
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.gates(combined)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c_next = forgetgate * c_cur + ingate * cellgate
        h_next = outgate * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_window, hidden_dims, kernel_size, output_window, output_dim = (448,304)):
        super(ConvLSTM, self).__init__()
        self.input_window = input_window
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.output_window = output_window 
        self.output_dim = output_dim 

        self.conv_lstm = ConvLSTMCell(1, hidden_dims[0], kernel_size)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_size, padding=kernel_size//2)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_size, padding=kernel_size//2)

        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim[0] * output_dim[1] * output_window)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, L, H, W = x.shape

        h, c = torch.zeros(B, self.hidden_dims[0], H, W, device=x.device), torch.zeros(B, self.hidden_dims[0], H, W, device=x.device)

        for t in range(L):
            x_t = x[:, t, :, :].unsqueeze(1)
            h, c = self.conv_lstm(x_t, (h, c))

        x = self.maxpool1(h)
        x = F.relu(self.conv1(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv2(x))
        x = x.view(B, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        output = x.view(B, self.output_window, self.output_dim[0], self.output_dim[1])
        output = self.sigmoid(output)

        return output