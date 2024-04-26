# Readme
___
This repository is the official implementation of **Unicorn: U-Net for Sea Ice Forecasting with Convolutional Neural
ODE** and benchmark models with pytorch.


<p align="center">
<img src="https://github.com/Optim-Lab/sif-models/assets/98927724/b3f93ae8-61f0-4dcf-8f87-37d1f1208a52" align="center" width="50%"/>
</p>


Unicorn is an innovative sea ice forecasting model that effectively captures the spatiotemporal dynamics of sea ice by integrating ConvNODE and time series decomposition within the U-Net framework, significantly enhancing predictions of sea ice concentration and extent over existing models.


<img src="https://github.com/Optim-Lab/sif-models/assets/98927724/59a4d96c-3d7b-40b7-bac1-aa612b16810f"/>



## Usage
___
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

## Citation
---