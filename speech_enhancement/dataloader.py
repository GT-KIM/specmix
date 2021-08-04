import torch
import os
import random
import librosa
import tqdm
from torch.utils.data import Dataset
from utils import *

class SignalDataset(Dataset) :
    def __init__(self, cleanpath, noisypath, training=False) :
        super(SignalDataset, self).__init__()
        self.cleanpath = cleanpath
        self.noisypath = noisypath
        self.sequence_length = 32768
        self.training = training
        self.datalist = sorted(os.listdir(self.cleanpath))
        if self.training :
            random.shuffle(self.datalist)
        self.sample_rate = 16000

        self.full_data = list()
        self.load_full_data()


    def load_full_data(self) :
        for idx in tqdm.trange(len(self.datalist)) :
            dataname = self.datalist[idx]
            clean, _ = librosa.load(self.cleanpath + dataname, sr=self.sample_rate)
            noisy, _ = librosa.load(self.noisypath + dataname, sr=self.sample_rate)

            length = len(clean)
            if self.training :
                if not length % self.sequence_length == 0 :
                    clean = np.pad(clean, (0, self.sequence_length - (length % self.sequence_length)), 'constant', constant_values=0)
                    noisy = np.pad(noisy, (0, self.sequence_length - (length % self.sequence_length)), 'constant', constant_values=0)

                end = self.sequence_length
                while end < len(clean) :
                    self.full_data.append({'clean': clean[end-self.sequence_length : end], 'noisy': noisy[end-self.sequence_length : end], 'length': self.sequence_length})
                    end += int(self.sequence_length * 0.5)
            else :
                self.full_data.append({'idx' : idx, 'clean' : clean, 'noisy' : noisy, 'length' : length})

    def __len__(self) :
        return len(self.full_data)

    def __getitem__(self, idx) :
        clean = self.full_data[idx]['clean']
        noisy = self.full_data[idx]['noisy']
        length =self.full_data[idx]['length']

        return {'clean' : clean, 'noisy' : noisy, 'length' : length}

def get_spec(wav) :
    wav_pad = librosa.util.fix_length(wav, len(wav) + 512 // 2)
    return librosa.stft(wav_pad, n_fft=512, hop_length=int(16000 * 0.016),
                              win_length=int(16000 * 0.032), window='hann')

def get_wav(spec, length) :
    return librosa.istft(spec[...,0] + 1j * spec[..., 1], hop_length=int(16000 * 0.016),
                              win_length=int(16000 * 0.032), window='hann', length=length)

def get_mag(spec) :
    mag, phase = librosa.magphase(spec)
    mean = np.mean(mag, axis=1).reshape((257,1))
    std = np.std(mag, axis=1).reshape((257,1)) + 1e-12
    mag = (mag - mean) / std
    return mag

def collate_fn(batch) :
    clean = [x['clean'] for x in batch]
    noisy = [x['noisy'] for x in batch]
    length = [x['length'] for x in batch]

    clean_spec = [get_spec(x) for x in clean]
    noisy_spec = [get_spec(x) for x in noisy]

    clean = torch.tensor(clean)
    noisy = torch.tensor(noisy)

    real = torch.from_numpy(np.real(clean_spec))
    imag = torch.from_numpy(np.imag(clean_spec))
    clean_spec = torch.stack((real, imag), dim=1)

    real = torch.from_numpy(np.real(noisy_spec))
    imag = torch.from_numpy(np.imag(noisy_spec))
    noisy_spec = torch.stack((real, imag), dim=1)

    return clean.to('cuda'), noisy.to('cuda'), clean_spec.to('cuda'), noisy_spec.to('cuda'), length
