import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from librosa.display import specshow

def plot_train(clean, estimated, noisy, name='result') :
    clean = clean.squeeze(0).cpu().numpy()
    estimated = estimated.squeeze(0).cpu().detach().numpy()
    noisy = noisy.squeeze(0).cpu().numpy()
    clean = np.sqrt(clean[0,:,:] ** 2 +  clean[1,:,:] **2)
    estimated = np.sqrt(estimated[0,:,:] ** 2 +  estimated[1,:,:] **2)
    noisy = np.sqrt(noisy[0,:,:] ** 2 +  noisy[1,:,:] **2)

    specshow(np.log(clean), cmap=cm.jet)
    plt.savefig("./plots/"+name+"_spec.png", dpi=300)
    plt.clf()
    specshow(np.log(estimated), cmap=cm.jet)
    plt.savefig("./plots/"+name+"_spec1.png", dpi=300)
    plt.clf()
    specshow(np.log(noisy), cmap=cm.jet)
    plt.savefig("./plots/"+name+"_spec2.png", dpi=300)
    plt.clf()

    plt.subplot(311)
    specshow(np.log(clean), cmap=cm.jet)
    plt.subplot(312)
    specshow(np.log(estimated), cmap=cm.jet)
    plt.subplot(313)
    specshow(np.log(noisy), cmap=cm.jet)
    plt.show()


def plot_data(clean, estimated, noisy) :
    clean = clean.squeeze(0).cpu().numpy()
    estimated = estimated.squeeze(0).detach().cpu().numpy()
    noisy = noisy.squeeze(0).cpu().numpy()
    #clean = np.sqrt(clean[...,0] ** 2 +  clean[...,1] **2)
    #estimated = np.sqrt(estimated[...,0] ** 2 +  estimated[...,1] **2)
    #noisy = np.sqrt(noisy[...,0] ** 2 +  noisy[...,1] **2)

    plt.subplot(311)
    specshow(np.log(clean), cmap=cm.jet)
    plt.subplot(312)
    specshow(np.log(estimated), cmap=cm.jet)
    plt.subplot(313)
    specshow(np.log(noisy), cmap=cm.jet)
    plt.show()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def masking(spec, ratio=np.random.rand(1)[0]) :
    mask = np.ones(spec.shape)

    t_times = np.random.randint(3)
    f_times = np.random.randint(3)

    for _ in range(t_times) :
        t = np.random.randint((1-ratio)*mask.shape[0]+1)
        mask[t:t+int(mask.shape[0]*ratio), :] = 0

    for _ in range(f_times) :
        f = np.random.randint((1-ratio)*mask.shape[1]+1)
        mask[:, f:f+int(mask.shape[1]*ratio)] = 0
    inv_mask = -1 * (mask - 1)

    return mask, inv_mask