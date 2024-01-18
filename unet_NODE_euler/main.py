from preprocess import *
from model import *
from train import *
from eval import *
from util import *
from metric import *
from torch import optim
import matplotlib.pyplot as plt

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
    de = args.de
    pe = args.pe
    key = args.key
    name = args.name    

    seed_everything(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if key is not None:
        
        wandb.login(key=key)
        wandb.init(project='UnetNODE_euler', name=name)

    data = load_data()

    tr_ds, va_ds, te_ds = split_data(data)

    tr_dataset = WindowDataset(tr_ds, input_window, output_window, 1) 
    va_dataset = WindowDataset(va_ds, input_window, output_window, 1)
    te_dataset = WindowDataset(te_ds, input_window, output_window, 1) 

    tr_loader,va_loader,te_loader = loader(tr_dataset,va_dataset,te_dataset,batch_size)

    model = UnetNODE(input_window, output_window, de, pe).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss().to(device)
    best_valid_loss = float('inf')   

    with tqdm(range(1,  epochs+1)) as tr:
        for epoch in tr:

            train_loss = train(model, tr_loader, optimizer, criterion, device)
            pred,target,valid_loss = eval(model, va_loader, criterion, device)
            
            
            plt.imshow(pred[-1])
            plt.savefig('pred.png')

            plt.imshow(target[-1])
            plt.savefig('target.png')
            
            if key is not None:
                # l1_1000, l2_1000, ssim_score, ms_ssim_score, lpips_score, iiee_score = get_score(pred, target, output_window, device)
                wandb.log({
                    "Predict Image": [
                        wandb.Image('pred.png')
                    ],
                    "Target Image": [
                        wandb.Image('target.png')
                    ],
                    'epoch' : epoch,
                    'train_loss': train_loss,
                    'valid_loss' : valid_loss,
                    "mae_score" : get_mae_score(target,pred), 
                    "f1_score" : get_f1_score(target,pred),
                    "mae_over_f1" : get_mae_over_f1(target,pred)
                    # '1000l1': l1_1000,
                    # '1000l2': l2_1000,
                    # 'ssim_score': ssim_score,
                    # 'ms_ssim_score': ms_ssim_score,
                    # 'lpips_score': lpips_score,
                    # 'iiee_score': iiee_score
                })
            
            if epoch % 10 == 0:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_unet_NODE_euler.pth')
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                
                # 조기 종료 체크
            if early_stopping_count >= early_stopping:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
                print(f'best valid loss :{best_valid_loss}')
                break

    model = UnetNODE(input_window, output_window, de, pe).to(device)
    model.load_state_dict(torch.load('best_unet_NODE_euler.pth'))
    pred, target, loss = eval(model, te_loader, criterion, device)
    
    mae_score = get_mae_score(target, pred)
    f1_score = get_f1_score(target, pred)
    mae_over_f1  = get_mae_over_f1(target, pred)
    #l1_1000, l2_1000, ssim_score, ms_ssim_score, lpips_score, iiee_score = get_score(pred, target, output_window, device)

    if key is not None:
        table = wandb.Table(columns=["seed","learning_rate","best_model_loss","mae_score", "f1_score","mae_over_f1"])
        table.add_data(seed,learning_rate,loss,mae_score,f1_score,mae_over_f1)
        wandb.log({
        # 'best_model_loss': loss,
        # '1000l1': l1_1000,
        # '1000l2': l2_1000,
        # 'ssim_score': ssim_score,
        # 'ms_ssim_score': ms_ssim_score,
        # 'ms_ssim_score': lpips_score,
        # 'iiee_score': iiee_score,
        # 'seed': seed,
        # 'learning_rate': learning_rate,
        "metrics_table": table
        })

if __name__ == "__main__":
    main()