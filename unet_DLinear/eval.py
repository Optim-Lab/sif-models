#%%
import numpy as np
import torch

#%%
def eval(model, data_loader, criterion, device):
    model.eval()
    
    total_loss=[]
    predictions = []
    
    with torch.no_grad():
        for i,(input,target) in enumerate(data_loader):
            

            input = input.to(device)
            target = target.to(device)
            
            pred = model(input)
            loss = criterion(pred, target)
            
            total_loss.append(loss)
            predictions.append(pred)
        
            
            

    predict = np.array(predictions[-1][0].cpu())
    target = np.array(target[-1].cpu())
    out_loss = sum(total_loss) / len(total_loss)
    return predict, target, out_loss