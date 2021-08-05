import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
import librosa
import soundfile as sound


import torch
from torch import nn, optim

from utils import MixupGenerator, CutmixGenerator, SpecmixGenerator, SpecaugmentGenerator, EnergymaskingGenerator
from focal_loss import Focal_loss, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
MODE = 'DEV'
ThisPath = 'data/DCASE2020A/'
num_audio_channels = 1
sr = 44100

if MODE == 'DEV':
    TrainFile = ThisPath + 'evaluation_setup/fold1_train.csv'
    ValFile = ThisPath + 'evaluation_setup/fold1_evaluate.csv'

SampleDuration = 10

#log-mel spectrogram parameters
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/float(HopLength)))

multi_gpus = 1    # the number of GPUs used for training

#training parameters
data_augment = 'Energymasking'
max_lr = 0.001
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

def to_categorical(y, num_classes) :
    return np.eye(num_classes, dtype='uint8')[y]

#load filenames and labels
dev_train_df = pd.read_csv(TrainFile,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)
y_train = to_categorical(y_train_labels, NumClasses)

if MODE == 'DEV':
    dev_val_df = pd.read_csv(ValFile,sep='\t', encoding='ASCII')
    wavpaths_val = dev_val_df['filename'].tolist()
    y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values
    y_val = to_categorical(y_val_labels, NumClasses)

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

    if i % 1500 == 1499:
        print("%i/%i training samples done" % (i + 1, len(wavpaths_train)))

print("Done")

LM_train = np.log(LM_train + 1e-8)

if delta:
    LM_deltas_train = deltas(LM_train)
    LM_deltas_deltas_train = deltas(LM_deltas_train)
    LM_train = np.concatenate((LM_train[:, :, 4:-4, :], LM_deltas_train[:, :, 2:-2, :], LM_deltas_deltas_train),
                              axis=-1)

if MODE == 'DEV':
    LM_val = np.zeros((len(wavpaths_val), NumFreqBins, NumTimeBins, num_audio_channels), 'float32')
    for i in range(len(wavpaths_val)):
        sig, fs = sound.read(ThisPath + wavpaths_val[i], stop=SampleDuration * sr)
        for channel in range(num_audio_channels):
            if len(sig.shape) == 1:
                sig = np.expand_dims(sig, -1)
            LM_val[i, :, :, channel] = librosa.feature.melspectrogram(sig[:, channel],
                                                                      sr=sr,
                                                                      n_fft=NumFFTPoints,
                                                                      hop_length=HopLength,
                                                                      n_mels=NumFreqBins,
                                                                      fmin=0.0,
                                                                      fmax=sr / 2,
                                                                      htk=True,
                                                                      norm=None)
        if i % 700 == 699:
            print("%i/%i val samples done" % (i + 1, len(wavpaths_val)))

    print("Done")
    LM_val = np.log(LM_val + 1e-8)
    if delta:
        LM_deltas_val = deltas(LM_val)
        LM_deltas_deltas_val = deltas(LM_deltas_val)
        LM_val = np.concatenate((LM_val[:, :, 4:-4, :], LM_deltas_val[:, :, 2:-2, :], LM_deltas_deltas_val), axis=-1)

if delta:
    num_audio_channels *= 3

print('training data dimension: ', LM_train.shape)
if MODE == 'DEV':
    print('validation data dimension: ', LM_val.shape)

print('training labels dimension: ', y_train.shape)
if MODE == 'DEV':
    print('validation labels dimension: ', y_val.shape)

# create and compile the model
class trainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if crop_length > 0 :
            self.x = self.x[:, :, 0:crop_length, :]
        else :
            self.x = self.x

        return self.x[idx], self.y[idx]

class testDataset(Dataset) :
    def __init__(self, x,y) :
        self.x = x
        self.y = y

    def __len__(self) :
        return len(self.x)

    def __getitem__(self, idx) :
        return self.x[idx], self.y[idx]

TestData = testDataset(LM_val, y_val)
TestDataLoader = DataLoader(TestData, batch_size=1, shuffle=False)

from model.baseline import model_resnet
model = models.resnet101(pretrained=False)
model.fc = nn.Linear(2048, NumClasses)
"""
model = model_resnet(NumClasses,
                     num_filters=num_filters,
                     num_stacks=num_stacks,
                     output_num_filter_factor=output_num_filters_factor,
                     stacking_frame=stacking_frames,
                     domain_aux=False)
"""
optimizer = optim.Adam(model.parameters(),lr=max_lr,weight_decay=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30,50, 70,90], gamma=0.1)
criterion = CrossEntropyLoss#Focal_loss()

savedir = "ckpt/"
if not os.path.isdir(savedir) :
    os.makedirs(savedir)

# create data generator
if data_augment == 'None' :
    TrainData = trainDataset(LM_train, y_train)
    TrainDataGen = DataLoader(TrainData, batch_size=batch_size, shuffle=True)
elif data_augment == 'Mixup' :
    TrainDataGen = MixupGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()
elif data_augment == 'Cutmix' :
    TrainDataGen = CutmixGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()

elif data_augment == 'Specaugment' :
    TrainDataGen = SpecaugmentGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()
elif data_augment == 'Specmix' :
    TrainDataGen = SpecmixGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()

elif data_augment == 'Energymasking' :
    TrainDataGen = EnergymaskingGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()
model.cuda()

model.train()
epoch = 0
if data_augment == 'None' :
    idx = 0
    loss_list = list()
    acc_list = list()
    best_acc = 0.
    for i_epoch in range(10000):
        for i, data in enumerate(TrainDataGen) :
            model.train()
            X, Y = data
            X = X.permute(0, 3, 1, 2)
            X = X.float().cuda()
            Y = Y.float().cuda()
            pred = model(X)

            loss = criterion(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            idx += 1
            loss_list.append(loss.detach().cpu().numpy())

            if idx % 1000 == 0 :
                print("iter %d | loss %f"%(idx, loss.detach().cpu().numpy()))
                scheduler.step()
                state = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                if not os.path.isdir("ckpt_{}/".format(data_augment)) :
                    os.makedirs("ckpt_{}/".format(data_augment))
                #torch.save(state, "ckpt_{}/model-{}.pth".format(data_augment, epoch))
                epoch += 1

                model.eval()
                y_pred = list()
                y_gt = list()
                for i_test, data in enumerate(TestDataLoader):
                    X, Y = data
                    X = X.permute(0, 3, 1, 2)
                    X = X.float().cuda()
                    pred = model(X)
                    gt = np.argmax(Y.numpy(), axis=1)
                    pred = np.argmax(pred.detach().cpu().numpy())
                    y_gt.append(gt)
                    y_pred.append(pred)
                y_gt = np.array(np.reshape(y_gt, (-1)))
                y_pred = np.array(y_pred)
                Overall_accuracy = np.sum(y_gt == y_pred) / float(LM_val.shape[0])
                print("overall test accuracy: ", Overall_accuracy)
                acc_list.append(Overall_accuracy)
                if Overall_accuracy >= best_acc :
                    best_acc = Overall_accuracy
                    state = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    if not os.path.isdir("ckpt_{}/".format(data_augment)):
                        os.makedirs("ckpt_{}/".format(data_augment))
                    torch.save(state, "ckpt_{}/model.pth".format(data_augment))

            if idx == 200000 :
                print("Train Finished!")
                if not os.path.isdir("result_{}/".format(data_augment)) :
                    os.makedirs("result_{}/".format(data_augment))
                np.savez("result_{}/curve.npz".format(data_augment), loss=loss_list, acc=acc_list, best_acc = best_acc)
                break
else :
    loss_list = list()
    acc_list = list()
    best_acc = 0.
    for idx, data in enumerate(TrainDataGen):
        model.train()
        X, Y = data
        X = np.transpose(X, (0, 3, 1, 2))
        X = torch.from_numpy(X).float().cuda()
        Y = torch.from_numpy(Y).float().cuda()
        pred = model(X)

        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().numpy())

        if idx % 1000 == 0:
            print("iter %d | loss %f" % (idx, loss.detach().cpu().numpy()))
            scheduler.step()
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            if not os.path.isdir("ckpt_{}/".format(data_augment)):
                os.makedirs("ckpt_{}/".format(data_augment))
            #torch.save(state, "ckpt_{}/model-{}.pth".format(data_augment, epoch))
            epoch += 1

            model.eval()
            y_pred = list()
            y_gt = list()
            for i_test, data in enumerate(TestDataLoader):
                X, Y = data
                X = X.permute(0, 3, 1, 2)
                X = X.float().cuda()
                pred = model(X)
                gt = np.argmax(Y.numpy(), axis=1)
                pred = np.argmax(pred.detach().cpu().numpy())
                y_gt.append(gt)
                y_pred.append(pred)
            y_gt = np.array(np.reshape(y_gt, (-1)))
            y_pred = np.array(y_pred)
            Overall_accuracy = np.sum(y_gt == y_pred) / float(LM_val.shape[0])
            print("overall test accuracy: ", Overall_accuracy)
            acc_list.append(Overall_accuracy)
            if Overall_accuracy >= best_acc:
                best_acc = Overall_accuracy
                state = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                if not os.path.isdir("ckpt_{}/".format(data_augment)):
                    os.makedirs("ckpt_{}/".format(data_augment))
                torch.save(state, "ckpt_{}/model.pth".format(data_augment))

        if idx == 200000:
            print("Train Finished!")
            if not os.path.isdir("result_{}/".format(data_augment)):
                os.makedirs("result_{}/".format(data_augment))
            np.savez("result_{}/curve.npz".format(data_augment), loss=loss_list, acc=acc_list, best_acc=best_acc)
            break