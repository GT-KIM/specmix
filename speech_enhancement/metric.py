
import torch
import torch.functional as F

from pesq import pesq


class PESQ:
    def __init__(self):
        self.pesq = pesq_metric

    def __call__(self, output, bd):
        return self.pesq(output, bd)


def pesq_metric(y_hat, bd):
    # PESQ
    with torch.no_grad():
        #y_hat = y_hat.cpu().numpy()
        y = bd#bd.cpu().numpy()  # target signal

        sum = 0
        for i in range(len(y)):
            sum += pesq(16000,y[i], y_hat[i], 'wb')

        sum /= len(y)
        return torch.tensor(sum)