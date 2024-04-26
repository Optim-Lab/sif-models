# Readme
---
This repository is the official implementation of **Unicorn: U-Net for Sea Ice Forecasting with Convolutional Neural
ODE** and benchmark models with pytorch.

<img width="50">![unicorn](https://github.com/Optim-Lab/sif-models/assets/98927724/7ee94980-5432-4d63-9bea-d934f7f87089)

## Usage
---
### 1. proposed model
```
python unicorn_cv/main.py
```

### 2. benchmark models
```
python unet_cv/main.py
python dunet_cv/main.py
python nunet_cv/main.py
python CNN_cv/main.py
python ConvLSTM_cv/main.py
python sicnet_tsam_cv/main.py
python sicnet_cbam_cv/main.py
```

### 3. ablaition study models
```
python ablation_study_data_cv/main.py
python ablation_study_DCMP_cv/main.py
python ablation_study_ConvNODE_cv/main.py
```
