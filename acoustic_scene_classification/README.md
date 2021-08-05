# specmix acoustic scene classification
Pytorch implementation 

## Dependencies
Python==3.7  
Numpy==1.19.5  
Scipy==1.4.1  
Pytorch==1.7.1+cu101   
librosa==0.7.2  
matplotlib==3.2.1  
scikit-learn==0.23.1  
pandas==1.0.4

## Train

ThisPath = '/your/DCASE2020/data/path/'  
TrainFile = '/your/DCASE2020/data/path/fold1_train.csv'  
ValFile = '/your//DCASE2020/path/fold1_evaluate.csv'  

### train with No augmentation, Mixup, Cutmix, Specaugment, Specmix
data_augment = 'None' 'Mixup' or 'Cutmix' or 'Specaugment' or 'Specmix'  
> python train.py

## Evaluate
### Inference
data_augment = 'None' or 'Mixup' or 'Cutmix' or 'Specaugment' or 'Specmix'  
> python inference.py
### plot loss curve
> python test.py
## models are available here.

https://drive.google.com/drive/folders/1Rgqx9T591J-ZAo8eK9loNrvnRzz5-jM7?usp=sharing
