#%%
import lpips
from IQA_pytorch import SSIM, MS_SSIM, utils
from ignite.metrics import PSNR
from torchvision.utils import save_image
import torch
from PIL import Image
import torch.nn.functional as F

#%%

def get_score(pred, target, output_window, device):
    pred = torch.tensor(pred).to(device)
    target = torch.tensor(target).to(device)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    ssim = SSIM(channels=3)
    ms_ssim = MS_SSIM(channels=3)

    for k in range(pred.shape[0]):
        save_image(target[k], './unet_long_plot/target/target_{}.png'.format(k))
        save_image(pred[k], './unet_long_plot/pred/pred_{}.png'.format(k))

    ssim_score = 0
    ms_ssim_score = 0
    lpips_score = 0    

    for k in range(output_window):
        
        pred_path  = './unet_long_plot/pred/pred_{}.png'.format(k)
        target_path = './unet_long_plot/target/target_{}.png'.format(k)

        pred = utils.prepare_image(Image.open(pred_path).convert("RGB")).to(device)
        target = utils.prepare_image(Image.open(target_path).convert("RGB")).to(device)
        
        ssim_score += ssim(pred, target, as_loss=False).item()
        ms_ssim_score += ms_ssim(pred, target, as_loss=False).item()
        lpips_score += loss_fn_alex(pred, target).item()


    l2 = F.mse_loss(pred,target)
    l1 = F.l1_loss(pred,target)
    ssim_score = ssim_score/output_window
    ms_ssim_score = ms_ssim_score/output_window
    lpips_score = lpips_score/output_window

    print("1000l1 : {}".format(1000 * l1))
    print("1000l2 : {}".format(1000 * l2))
    print('ssim_score: %.4f' % ssim_score)
    print('ms_ssim_score: %.4f' % ms_ssim_score)
    print('lpips_score: %.4f' % lpips_score)

    return 1000*l1, 1000*l2, ssim_score, ms_ssim_score, lpips_score