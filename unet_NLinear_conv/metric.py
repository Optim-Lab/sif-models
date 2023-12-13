#%%
import lpips
from IQA_pytorch import SSIM, MS_SSIM, utils
from ignite.metrics import PSNR
from torchvision.utils import save_image
import torch
from PIL import Image
import torch.nn.functional as F
#%%
def calculate_IIEE(y_true, y_pred_proba, threshold=0, grid_cell_area=None):
    """
    Calculate the Integrated Ice Edge Error (IIEE).

    Parameters:
    y_true (Tensor): 실제 해빙 농도 (continuous values).
    y_pred_proba (Tensor): 예측된 해빙 농도 확률 (continuous values).
    threshold (float): 해빙 농도의 임계값 (default: 0).
    grid_cell_area (Tensor): 각 그리드 셀의 면적.

    Returns:
    Tensor: IIEE 값.
    """
    y_pred = (y_pred_proba > threshold).float()

    y_true_binary = (y_true > threshold).float()

    binary_error = torch.abs(y_true_binary - y_pred)

    if grid_cell_area is not None:
        IIEE = torch.sum(binary_error * grid_cell_area)
    else:
        IIEE = torch.sum(binary_error)

    return IIEE
#%%

def get_score(pred, target, output_window, device):
    pred = torch.tensor(pred).to(device)
    target = torch.tensor(target).to(device)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    ssim = SSIM(channels=3)
    ms_ssim = MS_SSIM(channels=3)

    for k in range(pred.shape[0]):
        save_image(target[k], './unet_nlinear_conv_plot/target/target_{}.png'.format(k))
        save_image(pred[k], './unet_nlinear_conv_plot/pred/pred_{}.png'.format(k))

    ssim_score = 0
    ms_ssim_score = 0
    lpips_score = 0    
    iiee_score = 0 
    

    for k in range(output_window):
        
        pred_path  = './unet_nlinear_conv_plot/pred/pred_{}.png'.format(k)
        target_path = './unet_nlinear_conv_plot/target/target_{}.png'.format(k)

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
    iiee_score = iiee_score/output_window/448/304

    print("1000l1 : {}".format(1000 * l1))
    print("1000l2 : {}".format(1000 * l2))
    print('ssim_score: %.4f' % ssim_score)
    print('ms_ssim_score: %.4f' % ms_ssim_score)
    print('lpips_score: %.4f' % lpips_score)
    print('iiee_score: %.4f' % iiee_score)

    return 1000*l1, 1000*l2, ssim_score, ms_ssim_score, lpips_score, iiee_score