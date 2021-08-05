import torch
import numpy as np
import threading, random

from librosa.display import specshow
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    # python 3
    def __next__(self):
        with self.lock:
            return self.it.__next__()

    # python 2
    #def next(self):
    #    with self.lock:
    #        return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

class MixupGenerator():
    '''
    Reference: https://github.com/yu4u/mixup-generator
    '''

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=False, crop_length=400, y_train_2=None,
                 datagen=None):  # datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.sample_num = len(X_train)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1, 0, 3, 2, 5, 4]

    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[int(i * self.batch_size * 2):int((i + 1) * self.batch_size * 2)]
                    # X, y = self.__data_generation(batch_ids)

                    # yield X, y
                    yield self.__data_generation(batch_ids)

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape((self.batch_size, 1, 1, 1))
        y_l = l.reshape((self.batch_size, 1))

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]

        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
                StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)

                X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
                X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]

                if X1.shape[-1] == 6:
                    # randomly swap left and right channels
                    if np.random.randint(2) == 1:
                        X1[j, :, :, :] = X1[j:j + 1, :, :, self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j, :, :, :] = X2[j:j + 1, :, :, self.swap_inds]

            X1 = X1[:, :, 0:self.NewLength, :]
            X2 = X2[:, :, 0:self.NewLength, :]

        X = X1 * 0.5 + X2 * (1.0 - 0.5)

        import os
        if not os.path.isdir("./plots/") :
            os.makedirs("./plots/")
        specshow(X[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/mixup_X.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X1[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/mixup_X1.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X2[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/mixup_X2.png", dpi=300, pad_inches=0)
        plt.show()

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        if self.y_train_2 is None:
            return X, y
        else:
            return X, [y, self.y_train_2[batch_ids[:self.batch_size]]]

class CutmixGenerator():
    '''
    Reference: https://github.com/yu4u/mixup-generator
    '''

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=False, crop_length=400, y_train_2=None,
                 datagen=None):  # datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.sample_num = len(X_train)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1, 0, 3, 2, 5, 4]

    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[int(i * self.batch_size * 2):int((i + 1) * self.batch_size * 2)]
                    # X, y = self.__data_generation(batch_ids)

                    # yield X, y
                    yield self.__data_generation(batch_ids)

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha)
        X_l = l
        y_l = l

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]

        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
                StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)

                X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
                X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]

                if X1.shape[-1] == 6:
                    # randomly swap left and right channels
                    if np.random.randint(2) == 1:
                        X1[j, :, :, :] = X1[j:j + 1, :, :, self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j, :, :, :] = X2[j:j + 1, :, :, self.swap_inds]

            X1 = X1[:, :, 0:self.NewLength, :]
            X2 = X2[:, :, 0:self.NewLength, :]

        bbx1, bby1, bbx2, bby2 = rand_bbox(X1.shape, X_l)
        X = X1.copy()
        X[:, bbx1:bbx2, bby1:bby2, :] = X2[:, bbx1:bbx2, bby1:bby2, :]


        X3 = X1.copy()
        X3[:, bbx1:bbx2, bby1:bby2, :] = 0

        X4 = np.zeros(X1.shape)
        X4[:, bbx1:bbx2, bby1:bby2, :] = X2[:, bbx1:bbx2, bby1:bby2, :]
        import os
        if not os.path.isdir("./plots/") :
            os.makedirs("./plots/")
        specshow(X[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/cutmix_X.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X1[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/cutmix_X1.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X2[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/cutmix_X2.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X3[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/cutmix_X1_mask.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X4[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/cutmix_X2_mask.png", dpi=300, pad_inches=0)

        plt.show()

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.shape[1] * X.shape[2]))
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * lam + y2 * (1.0 - lam))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * lam + y2 * (1.0 - lam)

        if self.y_train_2 is None:
            return X, y
        else:
            return X, [y, self.y_train_2[batch_ids[:self.batch_size]]]

class SpecmixGenerator():
    '''
    Reference: https://github.com/yu4u/mixup-generator
    '''

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=False, crop_length=400, y_train_2=None,
                 datagen=None):  # datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.sample_num = len(X_train)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1, 0, 3, 2, 5, 4]

    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[int(i * self.batch_size * 2):int((i + 1) * self.batch_size * 2)]
                    # X, y = self.__data_generation(batch_ids)

                    # yield X, y
                    yield self.__data_generation(batch_ids)

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha)
        X_l = l
        y_l = l

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]

        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
                StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)

                X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
                X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]

                if X1.shape[-1] == 6:
                    # randomly swap left and right channels
                    if np.random.randint(2) == 1:
                        X1[j, :, :, :] = X1[j:j + 1, :, :, self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j, :, :, :] = X2[j:j + 1, :, :, self.swap_inds]

            X1 = X1[:, :, 0:self.NewLength, :]
            X2 = X2[:, :, 0:self.NewLength, :]


        X = np.zeros(X1.shape)
        X3 = np.zeros(X1.shape)
        X4 = np.zeros(X1.shape)
        lam = list()
        for i in range(X1.shape[0]) :
            mask, inv_mask = masking(X1[0, :, :, 0])
            lam.append(mask.sum() / (X.shape[1] * X.shape[2]))
            for j in range(X1.shape[3]) :
                X[i,:,:,j] = mask * X1[i,:,:,j] + inv_mask * X2[i,:,:,j]

                X3[i,:,:,j] = mask * X1[i,:,:,j]
                X4[i,:,:,j] = inv_mask * X2[i,:,:,j]

        lam = np.array(lam)
        lam = lam.reshape((self.batch_size, 1))


        import os
        if not os.path.isdir("./plots/") :
            os.makedirs("./plots/")
        specshow(X[0, :, :, 0], cmap= cm.jet)
        plt.savefig("./plots/specmix_X.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X1[0, :, :, 0], cmap= cm.jet)
        plt.savefig("./plots/specmix_X1.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X2[0, :, :, 0], cmap= cm.jet)
        plt.savefig("./plots/specmix_X2.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X3[0, :, :, 0], cmap= cm.jet)
        plt.savefig("./plots/specmix_X1_mask.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X4[0, :, :, 0], cmap= cm.jet)
        plt.savefig("./plots/specmix_X2_mask.png", dpi=300, pad_inches=0)

        plt.show()

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * lam + y2 * (1.0 - lam))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * lam + y2 * (1.0 - lam)

        if self.y_train_2 is None:
            return X, y
        else:
            return X, [y, self.y_train_2[batch_ids[:self.batch_size]]]

class SpecaugmentGenerator():
    '''
    Reference: https://github.com/yu4u/mixup-generator
    '''

    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=False, crop_length=400, y_train_2=None,
                 datagen=None):  # datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_2 = y_train_2
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.sample_num = len(X_train)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1, 0, 3, 2, 5, 4]

    def __iter__(self):
        return self

    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[int(i * self.batch_size * 2):int((i + 1) * self.batch_size * 2)]
                    # X, y = self.__data_generation(batch_ids)

                    # yield X, y
                    yield self.__data_generation(batch_ids)

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha)
        X_l = l
        y_l = l

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]

        if self.NewLength > 0:
            for j in range(X1.shape[0]):
                StartLoc1 = np.random.randint(0, X1.shape[2] - self.NewLength)
                StartLoc2 = np.random.randint(0, X2.shape[2] - self.NewLength)

                X1[j, :, 0:self.NewLength, :] = X1[j, :, StartLoc1:StartLoc1 + self.NewLength, :]
                X2[j, :, 0:self.NewLength, :] = X2[j, :, StartLoc2:StartLoc2 + self.NewLength, :]

                if X1.shape[-1] == 6:
                    # randomly swap left and right channels
                    if np.random.randint(2) == 1:
                        X1[j, :, :, :] = X1[j:j + 1, :, :, self.swap_inds]
                    if np.random.randint(2) == 1:
                        X2[j, :, :, :] = X2[j:j + 1, :, :, self.swap_inds]

            X1 = X1[:, :, 0:self.NewLength, :]
            X2 = X2[:, :, 0:self.NewLength, :]


        X = np.zeros(X1.shape)
        lam = list()
        for i in range(X1.shape[0]) :
            mask, inv_mask = masking(X1[0, :, :, 0])
            lam.append(mask.sum() / (X.shape[1] * X.shape[2]))
            for j in range(X1.shape[3]) :
                X[i,:,:,j] = mask * X1[i,:,:,j]
        lam = np.array(lam)
        lam = lam.reshape((self.batch_size, 1))


        import os
        if not os.path.isdir("./plots/") :
            os.makedirs("./plots/")
        specshow(X[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/specaugment_X.png", dpi=300, pad_inches=0)
        plt.clf()
        specshow(X1[0, :, :, 0], cmap=cm.jet)
        plt.savefig("./plots/specaugment_X1.png", dpi=300, pad_inches=0)
        plt.show()


        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1)
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1

        if self.y_train_2 is None:
            return X, y
        else:
            return X, [y, self.y_train_2[batch_ids[:self.batch_size]]]

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
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

def masking(spec, ratio=0.1) :
    mask = np.ones(spec.shape)

    t_times = np.random.randint(3)
    f_times = np.random.randint(3)

    for _ in range(1) :
        t = np.random.randint((1-ratio)*mask.shape[0])
        mask[:, t:t+int(mask.shape[0]*ratio)] = 0

    for _ in range(1) :
        f = np.random.randint((1-ratio)*mask.shape[1])
        mask[f:f+int(mask.shape[1]*ratio), :] = 0
    inv_mask = -1 * (mask - 1)

    return mask, inv_mask

# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_gt, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          png_name=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_gt, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(png_name)
    return