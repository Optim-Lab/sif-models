from preprocess import *
from model import *
from train import *
from eval import *
from util import *
from metric import *
from torch import optim

import wandb

def main():
    args = argparse_custom()
    seed = args.s
    epochs = args.e
    batch_size = args.b
    learning_rate = args.lr
    early_stopping = args.es
    input_window = args.iw
    output_window = args.ow
    key = args.key
    name = args.name  
    ci = args.ci

    seed_everything(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)

    if key is not None:
        wandb.login(key=key)
        wandb.init(project='Unet_NLinear_cv', name=name)



    if ci == 1:
        data, mask = load_data_cv1()
        tr_ds, va_ds, te_ds = split_data_cv1(data)
    elif ci == 2:
        data, mask = load_data_cv2()
        tr_ds, va_ds, te_ds = split_data_cv2(data)
    elif ci == 3:
        data, mask = load_data_cv3()
        tr_ds, va_ds, te_ds = split_data_cv3(data)
    elif ci == 4:
        data, mask = load_data_cv4()
        tr_ds, va_ds, te_ds = split_data_cv4(data)


    tr_dataset = WindowDataset(tr_ds, input_window, output_window, 1) 
    va_dataset = WindowDataset(va_ds, input_window, output_window, 1)
    te_dataset = WindowDataset(te_ds, input_window, output_window, 1) 

    tr_loader,va_loader,te_loader = loader(tr_dataset,va_dataset,te_dataset,batch_size)

    model = UnetNLinear(input_window, output_window).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)
    best_valid_loss = float('inf')   

    with tqdm(range(1,  epochs+1)) as tr:
        for epoch in tr:

            train_loss = train(model, tr_loader, optimizer, criterion, device)
            pred, target, valid_loss, mae_score, rmse_score, f1_score, iiee_score, iou_score = eval(model, va_loader, criterion, device, mask)
            
            
            if key is not None:
                wandb.log({
                    'epoch' : epoch,
                    'train_loss': train_loss,
                    'valid_loss' : valid_loss,
                    "mae_score" : mae_score,
                    "rmse_score" : rmse_score,
                    "f1_score" : f1_score,
                    "iiee_score" : iiee_score,
                    "iou_score" : iou_score,

                })
            
            if epoch % 10 == 0:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './saved_models/best_unet_Nlinear_cv_{}_MSE_{}.pth'.format(seed, ci))
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                
            if early_stopping_count >= early_stopping:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
                print(f'best valid loss :{best_valid_loss}')
                break

    model = UnetNLinear(input_window, output_window).to(device)
    model.load_state_dict(torch.load('./saved_models/best_unet_Nlinear_cv_{}_MSE_{}.pth'.format(seed, ci)))
    pred, target, loss, mae_score, rmse_score, f1_score, iiee_score, iou_score = eval(model, te_loader, criterion, device, mask)

    if key is not None:
        table = wandb.Table(columns=["seed","learning_rate","loss","mae_score", "rmse_score","f1_score","iiee_score","iou_score"])
        table.add_data(seed,learning_rate,loss,mae_score,rmse_score,f1_score,iiee_score,iou_score)
        wandb.log({
        "metrics_table": table
        })

if __name__ == "__main__":
    main()