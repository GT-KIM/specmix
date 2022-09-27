# specmix acoustic scene classification
Pytorch implementation  
If you wanna use data augmentation method only, see util.py and use Generator classes.  
Since the Generator classes are written by numpy, you can use it to both Pytorch and Tensorflow.

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

https://www.dropbox.com/sh/7iqv0htycluwx35/AADFG9yxUdLLZaUs0j3U8mzTa?dl=0
