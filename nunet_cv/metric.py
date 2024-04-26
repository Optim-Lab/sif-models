#%%
import numpy as np
#%%
def get_iiee_score(true, pred, mask, threshold=0.15, grid_cell_area=None):
    expanded_mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], true.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, true.shape[1], axis=1)

    pred_binary = (pred > threshold).astype(float)
    true_binary = (true > threshold).astype(float)

    binary_error = np.sum(np.abs(true_binary - pred_binary) * expanded_mask)

    if grid_cell_area is not None:
        IIEE = np.sum(binary_error * grid_cell_area)
    else:
        IIEE = np.sum(binary_error)

    return IIEE / 4

def get_mae_score(true, pred, mask):
    expanded_mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], true.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, true.shape[1], axis=1)

    error = np.abs(true - pred) * expanded_mask
    score = np.sum(error) / np.sum(expanded_mask)
    return score

def get_rmse_score(true, pred, mask):
    expanded_mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], true.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, true.shape[1], axis=1)
    squared_error = ((true - pred) ** 2) * mask
    score = np.sqrt(np.sum(squared_error) / np.sum(expanded_mask))
    return score

def get_f1_score(true, pred, mask):
    expanded_mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], true.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, true.shape[1], axis=1)

    true = np.where(true < 0.15, 0, 1)
    pred = np.where(pred < 0.15, 0, 1)

    TP = np.sum((true == 1) & (pred == 1) & expanded_mask)
    FP = np.sum((true == 0) & (pred == 1) & expanded_mask)
    FN = np.sum((true == 1) & (pred == 0) & expanded_mask)
    
    score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    return score

def get_iou_score(true, pred, mask):
    expanded_mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], true.shape[0], axis=0)
    expanded_mask = np.repeat(expanded_mask, true.shape[1], axis=1)

    true = np.where((true >= 0.15) & expanded_mask, 1., 0.)
    pred = np.where((pred >= 0.15) & expanded_mask, 1., 0.)

    true_idx,  = np.where(true.reshape(-1) == 1.)
    pred_idx,  = np.where(pred.reshape(-1) == 1.)

    score = round(len(set(true_idx).intersection(pred_idx))/len(set(true_idx).union(pred_idx)) * 100, 2)
    return score
