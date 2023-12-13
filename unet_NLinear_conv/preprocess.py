#%%
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

#%%
def load_data(default_path = '../'):
    
    data = pd.read_csv(default_path+'data_v2/weekly_train.csv')
    train_data_path = default_path+'data_v2/weekly_train/'+data.tail(100*12)['week_file_nm'].values

    data_list = []
    for p in tqdm(train_data_path):
        data_list.append(np.load(p))

    data_ = np.array(data_list)
    data_ = data_/250

    ice_data = data_[:,:,:,0] + (data_[:,:,:,1] > 0) + data_[:,:,:,4]
    coast_land_mask = ((data_[:,:,:,2] + data_[:,:,:,3]) == 0)

    masked_data = ice_data * coast_land_mask
    
    return masked_data

#%%
def split_data(data):
    train_set, test_set = train_test_split(data, test_size=0.2, shuffle=False)
    valid_set, test_set = train_test_split(test_set, test_size=0.5, shuffle=False)
    return train_set, valid_set,test_set

#%%
class WindowDataset(Dataset):
    def __init__(self, data, input_window, output_window,stride=1):
        
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = L - self.seq_len + 1
        data_tensor = torch.tensor(data)

        X = torch.zeros(num_samples, input_window,448, 304)
        y = torch.zeros(num_samples, output_window,448, 304)

        for i in range(num_samples):
            
            X[i,:] = data_tensor[i*stride : i*stride+input_window,:, :]
            y[i,:] = data_tensor[i*stride+input_window : i*stride+self.seq_len,:,:]

        self.x = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len
#%%
def loader(train_set, valid_set,test_set, batch_size=8):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, drop_last=False)

    return train_loader, valid_loader, test_loader