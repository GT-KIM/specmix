from torch import optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import matplotlib.pyplot as plt
from dataloader import *
from model import *
from metric import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_augment = 'None'
batch_size=12

train_clean = "data/train_clean/"
train_noisy = "data/train_noisy/"

if __name__ == "__main__" :

    trainset = SignalDataset(train_clean, train_noisy, training=True)
    trainloader = DataLoader(trainset,batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = Network()
    model.to('cuda')
    criterion = nn.MSELoss().to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10, 20, 30], gamma=0.1)

    epoch_loss = 0.
    print("Train start")
    for epoch in range(30) :
        model.train()
        for iter, (clean, noisy, clean_spec, noisy_spec, length) in enumerate(trainloader) :
            mask, output = model(noisy_spec)

            #plot_train(clean[0], output[0], noisy[0])
            loss = criterion(output, clean_spec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            print("epoch : %d [%d/%d] batch loss : %f" %(epoch, iter, len(trainloader),loss.detach().item()))

            if iter % 100 == 0 and iter != 0 :
                if not os.path.isdir("model_{}/".format(data_augment)):
                    os.makedirs("model_{}/".format(data_augment))
                state = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, "model_{}/model.pth".format(data_augment))
        scheduler.step()
        epoch_loss /= (iter + 1)
        print("epoch : %d epoch loss : %f"%(epoch, epoch_loss))

