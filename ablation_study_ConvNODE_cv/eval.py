#%%
import numpy as np
import torch
from metric import *
import pandas as pd

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
        for i,(x1, x2, x3, x4, target) in enumerate(data_loader):
            

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)

            target = target.to(device)
            
            pred = model(x1,x2,x3,x4)
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


#%%
def eval_season(model, data_loader, criterion, device, mask, ci):
    model.eval()
    
    maes = []
    rmses = []
    f1s = []
    iiees = []
    ious =[]


    with torch.no_grad():
        for i,(x1, x2, x3, x4, target) in enumerate(data_loader):
            

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)

            target = target.to(device)
            
            pred = model(x1,x2,x3,x4)

            for j in range(4):
                np.save('/home2/wkdso84/myubai/local_git/ablation_study_ConvNODE_cv/season_result_{ci}/{num1}_{num2}'.format(ci = ci, num1 = i, num2 = j), np.array(pred[0][j].cpu()))

            maes.append(get_mae_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            rmses.append(get_rmse_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            f1s.append(get_f1_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
            iiee = get_iiee_score(np.array(target.cpu()), np.array(pred.cpu()), mask)
            iiees.append(iiee)
            ious.append(get_iou_score(np.array(target.cpu()), np.array(pred.cpu()), mask))
    

    df = pd.DataFrame({
        'mae': maes,
        'rmse': rmses,
        'f1': f1s,
        'iiee': iiees,
        'iou': ious
    })


    csv_file = "/home2/wkdso84/myubai/local_git/ablation_study_ConvNODE_cv/season_metrics_{}.csv".format(ci)
    df.to_csv(csv_file, index=True)
