from torch import optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import matplotlib.pyplot as plt
from dataloader import *
from model import *
from metric import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_augment = 'Specmix'
batch_size= 6
alpha = 0.4


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
    for epoch in range(500) :
        model.train()
        for iter, ((clean1, noisy1, clean_spec1, noisy_spec1, length1), (clean2, noisy2, clean_spec2, noisy_spec2, length2)) in enumerate(zip(trainloader, trainloader)) :
            if data_augment == 'Mixup' :
                l = np.random.beta(alpha, alpha, clean_spec1.shape[0])
                X_l = torch.from_numpy(l.reshape((clean_spec1.shape[0], 1, 1, 1))).float().cuda()
                noisy_spec = noisy_spec1 * X_l + noisy_spec2 * (1 - X_l)
                clean_spec = clean_spec1 * X_l + clean_spec2 * (1 - X_l)
            elif data_augment == 'Cutmix' :
                l = np.random.beta(alpha, alpha)
                bbx1, bby1, bbx2, bby2 = rand_bbox(noisy_spec1.shape, l)
                noisy_spec = noisy_spec1.clone()
                noisy_spec[:, :, bbx1:bbx2, bby1:bby2] = noisy_spec2[:, :, bbx1:bbx2, bby1:bby2]
                clean_spec = clean_spec1.clone()
                clean_spec[:, :, bbx1:bbx2, bby1:bby2] = clean_spec2[:, :, bbx1:bbx2, bby1:bby2]
            elif data_augment == 'Specaugment' :
                noisy_spec = torch.zeros(noisy_spec1.shape).float().cuda()
                clean_spec = torch.zeros(clean_spec1.shape).float().cuda()
                for i in range(clean_spec1.shape[0]):
                    mask, inv_mask = masking(clean_spec1[0, 0, :, :], ratio=np.random.rand(1)[0])
                    mask = torch.from_numpy(mask).float().cuda()
                    inv_mask = torch.from_numpy(inv_mask).float().cuda()
                    for j in range(clean_spec1.shape[1]):
                        noisy_spec[i, j, :, :] = mask * noisy_spec1[i, j, :, :]
                        clean_spec[i, j, :, :] = mask * clean_spec1[i,j, :, :]
            elif data_augment == 'Specmix' :
                noisy_spec = torch.zeros(noisy_spec1.shape).float().cuda()
                clean_spec = torch.zeros(clean_spec1.shape).float().cuda()
                for i in range(clean_spec1.shape[0]):
                    mask, inv_mask = masking(clean_spec1[0, 0, :, :], ratio=np.random.rand(1)[0])
                    mask = torch.from_numpy(mask).float().cuda()
                    inv_mask = torch.from_numpy(inv_mask).float().cuda()
                    for j in range(clean_spec1.shape[1]):
                        noisy_spec[i, j, :, :] = mask * noisy_spec1[i, j, :, :] + inv_mask * noisy_spec2[i, j, :, :]
                        clean_spec[i, j, :, :] = mask * clean_spec1[i,j, :, :] + inv_mask * clean_spec2[i, j, :, :]

            mask, output = model(noisy_spec)

            #plot_train(noisy_spec[0], noisy_spec1[0], noisy_spec2[0],'noisy')
            #plot_train(clean_spec[0], clean_spec1[0], clean_spec2[0],'clean')
            #plot_train(clean_spec[0], output[0], noisy_spec[0])

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

