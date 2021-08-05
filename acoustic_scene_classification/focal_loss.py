import torch

class Focal_loss:
    def __init__(self):
        self.loss = categorical_focal_loss

    def __call__(self, output, gt):
        return self.loss(output, gt)

def categorical_focal_loss(y_true, y_pred):
    """
    Reference: https://github.com/umbertogriffo/focal-loss-keras

    Softmax version of focal loss.
            m
      FL = SUM -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
           c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    gamma = 2.
    alpha = .25

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= torch.sum(y_pred, dim=-1, keepdim=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = 1e-07
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * torch.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return torch.mean(torch.mean(loss, dim=1), dim=0)

def CrossEntropyLoss(input, target) :
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]