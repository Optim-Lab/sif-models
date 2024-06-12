# Readme

This repository is the official implementation of **Unicorn: U-Net for Sea Ice Forecasting with Convolutional Neural
ODE** and benchmark models with pytorch.


<p align="center">
<img src="https://github.com/Optim-Lab/sif-models/assets/98927724/b3f93ae8-61f0-4dcf-8f87-37d1f1208a52" align="center" width="40%"/>
</p>


**Unicorn** is an innovative sea ice forecasting model that effectively captures the spatiotemporal dynamics of sea ice by integrating ConvNODE and time series decomposition within the U-Net framework, significantly enhancing predictions of sea ice concentration and extent over existing models.


<img src="https://github.com/Optim-Lab/sif-models/assets/98927724/59a4d96c-3d7b-40b7-bac1-aa612b16810f"/>


## Dataset
### Download
Download data from one of the following links and unpack it into `data_V2`.
- [SIC data](https://dacon.io/competitions/official/235731/data) (permission required)
- Ancillary data 
    * [TB data](https://nsidc.org/data/nsidc-0001/versions/6)
    * [AGE data](https://nsidc.org/data/nsidc-0611/versions/4)

### Preprocess
Ancillary data needs to be preprocessed.

TB 
1. Out of the rioxarray file load 37V frequency data scaling pixel values multiplied with the scale factor of 0.1.
2. Where there is missing pixels interpolate pixels using nearest neighbor. 
3. Average the daily dataset of TB to match the weekly temporal resolution of our main SIC dataset.  

SIA 
1. Mask out land, and group sea ice pixels to two groups where sea ice aged 2 years or more is catergorized as multi-year ice and the rest first year ice.
2. Use reprojection to change spatial coordinates and resolution size to match our main SIC dataset.   


## Usage
The model can be trained using the codes below, and the results can be viewed with the options `-key "wandb key" -name "project name"`.

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
