# specmix speech enhancement task
Pytorch implementation 

![image1](./speech_enhancement/image/model.jpg)


## Dependencies
Python==3.7  
Numpy==1.19.5  
Scipy==1.4.1  
Pytorch==1.7.1+cu101  
oct2py==5.0.4  
pesq==0.0.1 (see https://github.com/ludlows/python-pesq)  
librosa==0.7.2  
matplotlib==3.2.1  
tqdm==4.46.1  
MATLAB(optional)  

## Train

train_clean='your/DEMANDdata/train_clean/.wav'  
train_noisy='your/DEMANDdata/train_noisy/.wav'  

### No augmentation
> python train.py

### Mixup, Cutmix, Specaugment, Specmix
data_augment = 'Mixup' or 'Cutmix' or 'Specaugment' or 'Specmix'  
> python train_augment.py

## Evaluate
### Generate test wav file and calculate PESQ, CSIG, CBAK, COVL, SSNR
data_augment = 'None' or 'Mixup' or 'Cutmix' or 'Specaugment' or 'Specmix'  
train_clean='your/DEMANDdata/test_clean/.wav'  
train_noisy='your/DEMANDdata/test_noisy/.wav' 
> python test.py
### calculate PESQ, CSIG, CBAK, COVL, SSNR using MATLAB
> ./eval/main.m
## Data, model and enhanced speech are available here.

https://drive.google.com/drive/folders/1Rgqx9T591J-ZAo8eK9loNrvnRzz5-jM7?usp=sharing
