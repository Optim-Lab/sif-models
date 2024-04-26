#%%
import numpy as np
import torch
from metric import *

#%%
def eval(model, data_loader, criterion, device, mask):
    model.eval()
    
    total_loss=[]
    predictions = []
    maes = []
    rmses = []
    f1s = []
    iiees = []
    ious =[]


    with torch.no_grad():
        for i,(input,target) in enumerate(data_loader):
            

            input = input.to(device)
            target = target.to(device)
            
            pred = model(input)
            loss = criterion(pred, target)
            
            total_loss.append(loss)
            predictions.append(pred)

            maes.append(get_mae_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            rmses.append(get_rmse_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            f1s.append(get_f1_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            iiee = get_iiee_score(np.array(target.cpu()), np.array(pred.cpu()), mask)
            iiees.append(iiee)
            ious.append(get_iou_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
        
            
            

    predict = np.array(predictions[-1][0].cpu())
    target = np.array(target[-1].cpu())
    out_loss = sum(total_loss) / len(total_loss)

    mae_score = sum(maes) / len(maes)
    rmse_score = sum(rmses) / len(rmses)
    f1_score = sum(f1s) / len(f1s)
    iiee_score = sum(iiees) / len(iiees)
    iou_score = sum(ious) / len(ious)

    return predict, target, out_loss , mae_score, rmse_score, f1_score, iiee_score, iou_score