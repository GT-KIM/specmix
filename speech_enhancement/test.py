import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch import optim
from torch.utils.data import DataLoader
import torch
from dataloader import *
from model import *
from metric import *
import soundfile as sf
from semetrics import *

test_clean = "data/test_clean/"
test_noisy = "data/test_noisy/"
data_augment = 'Specmix'

if __name__ == "__main__" :
    testset = SignalDataset(test_clean, test_noisy, training=False)
    testloader = DataLoader(testset,batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = Network()
    model.to('cuda')
    criterion = nn.MSELoss().to('cuda')
    state = torch.load("model_{}/model.pth".format(data_augment))
    model.load_state_dict(state['model'])
    epoch_loss = 0.
    epoch_pesq = 0.
    epoch_csig = 0.
    epoch_cbak = 0.
    epoch_covl = 0.
    epoch_ssnr = 0.
    epoch_pesq_noisy = 0.
    print("Evaluate start")

    model.eval()
    idx = 0
    with torch.no_grad() :
        for iter, (clean, noisy, clean_spec, noisy_spec, length) in enumerate(testloader) :
            mask, output = model(noisy_spec)

            #plot_train(clean_spec[0], output[0,:,:,:], noisy_spec[0])

            clean = clean_spec.permute(0, 2, 3, 1)
            output = output.permute(0, 2, 3, 1)
            noisy = noisy_spec.permute(0, 2, 3, 1)
            gt = get_wav(clean.squeeze(0).cpu().numpy(), length=length[0])[np.newaxis, :]
            pred = get_wav(output.squeeze(0).cpu().numpy(), length=length[0])[np.newaxis, :]
            noisy_gt = get_wav(noisy.squeeze(0).cpu().numpy(), length=length[0])[np.newaxis, :]

            if not os.path.isdir("eval/test_{}/clean/".format(data_augment)) :
                os.makedirs("eval/test_{}/clean/".format(data_augment))
            if not os.path.isdir("eval/test_{}/estimated/".format(data_augment)) :
                os.makedirs("eval/test_{}/estimated/".format(data_augment))
            if not os.path.isdir("eval/test_{}/noisy/".format(data_augment)):
                os.makedirs("eval/test_{}/noisy/".format(data_augment))

            for i in range(len(gt)) :
                gt[i] = np.clip(gt[i], -1, 1)
                pred[i] = np.clip(pred[i], -1, 1)
                noisy_gt[i] = np.clip(noisy_gt[i], -1, 1)
                sf.write("eval/test_{}/clean/{}.wav".format(data_augment, idx), gt[i], 16000)
                sf.write("eval/test_{}/estimated/{}.wav".format(data_augment, idx), pred[i], 16000)
                sf.write("eval/test_{}/noisy/{}.wav".format(data_augment, idx), noisy_gt[i], 16000)
                pesq, csig, cbak, covl, ssnr = composite("eval/test_{}/clean/{}.wav".format(data_augment,idx),
                                                     "eval/test_{}/estimated/{}.wav".format(data_augment,idx))
                #pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy = composite("eval/clean/{}.wav".format(idx),
                #                                     "eval/noisy/{}.wav".format(idx))

                print(idx)
                print('estimated : ', pesq, csig, cbak, covl, ssnr)
                #print('noisy : ',pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy)
                epoch_pesq += pesq
                epoch_csig += csig
                epoch_cbak += cbak
                epoch_covl += covl
                epoch_ssnr += ssnr
                idx += 1
                #plot_data(clean[i], mask[i], noisy[i])

        epoch_pesq /= idx
        epoch_csig /= idx
        epoch_cbak /= idx
        epoch_covl /= idx
        epoch_ssnr /= idx
        print("test epoch pesq : %f csig : %f cbak : %f covl : %f ssnr : %f"%(epoch_pesq, epoch_csig, epoch_cbak,epoch_covl, epoch_ssnr))
