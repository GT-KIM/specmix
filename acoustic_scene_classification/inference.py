import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import datetime
import pandas as pd
import librosa
import soundfile as sound
from sklearn.metrics import confusion_matrix
import sys, subprocess

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utils import MixupGenerator, plot_confusion_matrix
from focal_loss import Focal_loss

MODE = 'INF'  # 'DEV' uses the official data fold; 'VAL' uses all data in development set for training

ThisPath = 'data/DCASE2020A/'
num_audio_channels = 1
sr = 44100

if MODE == 'INF':
    TestFile = ThisPath + 'evaluation_setup/fold1_evaluate.csv'
    scene_map_str = """
    airport 0 
    bus 1
    metro 2
    metro_station 3
    park 4
    public_square 5
    shopping_mall 6
    street_pedestrian 7
    street_traffic 8
    tram 9
    """
SampleDuration = 10

#log-mel spectrogram parameters
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/float(HopLength)))

multi_gpus = 1    # the number of GPUs used for training

#training parameters
max_lr = 0.1
batch_size = 32
num_epochs = 310
mixup_alpha = 0.4
crop_length = 400 #
delta = True
num_filters = 28
output_num_filters_factor = 1
wd = 1e-3

num_stacks = 4    # number of residual stacks
stacking_frames = None # put None if not applied


'''
Applying domain adaptation OR using focal loss function
(Set TRUE for both flags is not supported)
'''

domain_aux = False     # whether to add an auxiliary classifier to apply mild domain adaptation
beta = 0.1            # apply weighting to this new loss

focal_loss = True    # whether to use focal loss
gamma=1.0
alpha=0.3

TEST = 1    #use 1/n data to verify the model before training; put 1 if not applied

assert((domain_aux and focal_loss) == False)

def to_categorical(y, num_classes) :
    return np.eye(num_classes, dtype='uint8')[y]

#load filenames and labels
dev_train_df = pd.read_csv(TestFile,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)
y_train = to_categorical(y_train_labels, NumClasses)

if domain_aux:
    y_train_domain_labels =  dev_train_df['source_label'].astype('category').cat.codes.values
    y_train_domain = to_categorical(y_train_domain_labels, 2)



# load wav files and get log-mel spectrograms, deltas, and delta-deltas
def deltas(X_in):
    X_out = (X_in[:, :, 2:, :] - X_in[:, :, :-2, :]) / 10.0
    X_out = X_out[:, :, 1:-1, :] + (X_in[:, :, 4:, :] - X_in[:, :, :-4, :]) / 5.0
    return X_out

LM_train = np.zeros((len(wavpaths_train), NumFreqBins, NumTimeBins, num_audio_channels), 'float32')
for i in range(len(wavpaths_train)):
    sig, fs = sound.read(ThisPath + wavpaths_train[i], stop=SampleDuration * sr)
    # print (sig.shape, fs)

    for channel in range(num_audio_channels):
        if len(sig.shape) == 1:
            sig = np.expand_dims(sig, -1)
        LM_train[i, :, :, channel] = librosa.feature.melspectrogram(sig[:, channel],
                                                                    sr=sr,
                                                                    n_fft=NumFFTPoints,
                                                                    hop_length=HopLength,
                                                                    n_mels=NumFreqBins,
                                                                    fmin=0.0,
                                                                    fmax=sr / 2,
                                                                    htk=True,
                                                                    norm=None)

    if i % 700 == 699:
        print("%i/%i test samples done" % (i + 1, len(wavpaths_train)))
print("Done")

LM_train = np.log(LM_train + 1e-8)

if delta:
    LM_deltas_train = deltas(LM_train)
    LM_deltas_deltas_train = deltas(LM_deltas_train)
    LM_train = np.concatenate((LM_train[:, :, 4:-4, :], LM_deltas_train[:, :, 2:-2, :], LM_deltas_deltas_train),
                              axis=-1)

if delta:
    num_audio_channels *= 3

print('test data dimension: ', LM_train.shape)
print('test labels dimension: ', y_train.shape)


# create and compile the model

from model.baseline import model_resnet
model = model_resnet(NumClasses,
                     num_filters=num_filters,
                     num_stacks=num_stacks,
                     output_num_filter_factor=output_num_filters_factor,
                     stacking_frame=stacking_frames,
                     domain_aux=domain_aux)

optimizer = optim.Adam(model.parameters(),lr=max_lr,weight_decay=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.1)
criterion = Focal_loss()

savedir = "ckpt/"
model.load_state_dict(torch.load("ckpt/model.pth")['model'])

# create data generator

class testDataset(Dataset) :
    def __init__(self, x,y) :
        self.x = x
        self.y = y

    def __len__(self) :
        return len(self.x)

    def __getitem__(self, idx) :
        return self.x[idx], self.y[idx]

TestData = testDataset(LM_train, y_train)
TestDataLoader = DataLoader(TestData, batch_size=1, shuffle=False)

model.cuda()
model.eval()
y_pred = list()
y_gt = list()
for i, data in enumerate(TestDataLoader) :
    X, Y = data
    X = X.permute(0,3,1,2)
    X = X.float().cuda()
    pred = model(X)
    gt = np.argmax(Y.numpy(), axis=1)
    pred = np.argmax(pred.detach().cpu().numpy())
    y_gt.append(gt)
    y_pred.append(pred)
y_gt = np.array(np.reshape(y_gt, (-1)))
y_pred = np.array(y_pred)
Overall_accuracy = np.sum(y_gt==y_pred)/float(LM_train.shape[0])
print("overall accuracy: ", Overall_accuracy)

plot_confusion_matrix(y_gt, y_pred,ClassNames, normalize=True, title=None, png_name="confusion_matrix")

conf_matrix = confusion_matrix(y_gt,y_pred)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)