#%%
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

# %%
def loader(train_set, valid_set, test_set, batch_size=8):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader

def load_data_cv1(default_path = './'):
    
    train_data_path = default_path+'data_v2/weekly_train/'
    data_list = []
    for p in tqdm(range(732)):
        data_list.append(np.load(train_data_path+'{}.npy'.format(11023+p)))

    data_ = np.array(data_list)
    data_ = data_/250

    ice_data = data_[:,:,:,0] + (data_[:,:,:,1] > 0) + data_[:,:,:,4]
    coast_land_mask = ((data_[:,:,:,2] + data_[:,:,:,3]) == 0)

    masked_data = ice_data * coast_land_mask
    
    return masked_data, coast_land_mask[0]

def split_data_cv1(data):
    train_set = data[:574,:,:]
    train_set = train_set[:-52,:,:]
    valid_set = train_set[-52:,:,:]
    test_set = data[574:,:,:]
    return train_set, valid_set ,test_set

def load_data_cv2(default_path = './'):
    
    train_data_path = default_path+'data_v2/weekly_train/'
    data_list = []
    for p in tqdm(range(731)):
        data_list.append(np.load(train_data_path+'{}.npy'.format(11180+p)))

    data_ = np.array(data_list)
    data_ = data_/250

    ice_data = data_[:,:,:,0] + (data_[:,:,:,1] > 0) + data_[:,:,:,4]
    coast_land_mask = ((data_[:,:,:,2] + data_[:,:,:,3]) == 0)

    masked_data = ice_data * coast_land_mask
    
    return masked_data, coast_land_mask[0]

def split_data_cv2(data):
    train_set = data[:575,:,:]
    train_set = train_set[:-52,:,:]
    valid_set = train_set[-52:,:,:]
    test_set = data[575:,:,:]
    return train_set, valid_set ,test_set

def load_data_cv3(default_path = './'):
    
    train_data_path = default_path+'data_v2/weekly_train/'
    data_list = []
    for p in tqdm(range(731)):
        data_list.append(np.load(train_data_path+'{}.npy'.format(11337+p)))

    data_ = np.array(data_list)
    data_ = data_/250

    ice_data = data_[:,:,:,0] + (data_[:,:,:,1] > 0) + data_[:,:,:,4]
    coast_land_mask = ((data_[:,:,:,2] + data_[:,:,:,3]) == 0)

    masked_data = ice_data * coast_land_mask
    
    return masked_data, coast_land_mask[0]

def split_data_cv3(data):
    train_set = data[:574,:,:]
    train_set = train_set[:-52,:,:]
    valid_set = train_set[-52:,:,:]
    test_set = data[574:,:,:]
    return train_set, valid_set ,test_set

def load_data_cv4(default_path = './'):
    
    train_data_path = default_path+'data_v2/weekly_train/'
    data_list = []
    for p in tqdm(range(733)):
        data_list.append(np.load(train_data_path+'{}.npy'.format(11493+p)))

    data_ = np.array(data_list)
    data_ = data_/250

    ice_data = data_[:,:,:,0] + (data_[:,:,:,1] > 0) + data_[:,:,:,4]
    coast_land_mask = ((data_[:,:,:,2] + data_[:,:,:,3]) == 0)

    masked_data = ice_data * coast_land_mask
    
    return masked_data, coast_land_mask[0]

def split_data_cv4(data):
    train_set = data[:575,:,:]
    train_set = train_set[:-52,:,:]
    valid_set = train_set[-52:,:,:]
    test_set = data[575:,:,:]
    return train_set, valid_set ,test_set