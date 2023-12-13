#%%
import random
import os
import numpy as np
import torch
import argparse
#%%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def argparse_custom():
    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("-s", type=int, default=[42], nargs=1, help="seed")
    parser.add_argument("-e", type=int, default=[200], nargs=1, help="epoch")
    parser.add_argument("-b", type=int, default=[8], nargs=1, help="batch_size")
    parser.add_argument("-lr", type=float, default=[1e-3], nargs=1, help="learning_rate")
    parser.add_argument("-es", type=int, default=[30], nargs=1, help="early_stopping")
    parser.add_argument("-iw", type=int, default=[12], nargs=1, help="input_window")
    parser.add_argument("-ow", type=int, default=[4], nargs=1, help="output_window")
    parser.add_argument("-de", type=int, default=[3], nargs=1, help="decomp")
    parser.add_argument("-key", type=str, default=[None], nargs=1, help="wandb login key")
    parser.add_argument("-name", type=str, default=['default'], nargs=1, help="wandb project run name")
    
    args = parser.parse_args()

    args.s = args.s[0]
    args.e = args.e[0]
    args.b = args.b[0]
    args.lr = args.lr[0]
    args.es = args.es[0]
    args.iw = args.iw[0]
    args.ow = args.ow[0]
    args.de = args.de[0]
    args.key = args.key[0]
    args.name = args.name[0]

    return args