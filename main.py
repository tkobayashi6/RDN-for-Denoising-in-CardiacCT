import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import option
from utils import SaveData
from data import MDE_Dataset
from model import rdn_denoise


def set_loss(args):
    lossType = args.loss_type
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction


def lr_scheduler(args, epoch, data_len, optimizer):
    if args.decay_type == 'RDN':
        lrDecay = 200
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif args.decay_type == 'GRDN':
        lrDecay = 2*10**5 // data_len
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(args, start_time, set_loss, lr_scheduler):
    savedata = SaveData(args)
    savedata.create_checkpoint_dir()
    savedata.write_args()

    # Dataloader
    data_train = MDE_Dataset(args, 'train')
    data_val = MDE_Dataset(args, 'val')
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    data_len = len(train_loader)

    # Device specification
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define the model
    model = rdn_denoise.RDN_denoise(args)
    model.to(device)
    if args.cudnn_benchmark:
        cudnn.benchmark = True
        
    # Optimizer
    optimizer = optim.Adam(model.parameters())

    # Loss
    loss_fn = set_loss(args)

    save_loss = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    max_epoch = args.epochs
    bestscore_train = 1000
    bestscore_val = 1000
    with tqdm(total=max_epoch) as pbar:
        for i in range(max_epoch):
            save_loss['epochs'].append(i+1)
            lr = lr_scheduler(args, i+1, data_len, optimizer)
            pbar.set_description('Epoch[{}/{}]'.format(i+1, max_epoch))

            # ---------- Training ----------
            batch_losses = []
            model.train()
            with tqdm(total=len(train_loader), leave=False) as pbar_train:
                for im, im_gt in train_loader:
                    pbar_train.set_description('Training batch progress')
                    im, im_gt = im.to(device), im_gt.to(device)
                    pred = model(im)
                    loss = loss_fn(pred, im_gt)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    pbar_train.update(1)
            epoch_loss = np.mean(batch_losses)
            save_loss['train_loss'].append(epoch_loss)
            save_loss['lr'].append(lr)

            # ---------- Validation ----------
            batch_losses_val = []
            model.eval()
            with torch.no_grad():
                with tqdm(total=len(val_loader), leave=False) as pbar_val:
                    for im, im_gt in val_loader:
                        pbar_val.set_description('Validation batch progress')
                        im, im_gt = im.to(device), im_gt.to(device)
                        pred = model(im)
                        val_loss = loss_fn(pred, im_gt)
                        batch_losses_val.append(val_loss.item())
                        pbar_val.update(1)
            epoch_loss_val = np.mean(batch_losses_val)
            save_loss['val_loss'].append(epoch_loss_val)

            log = '\nEpoch {}: {} Loss [train:{:.4f}/val:{:.4f}], lr={}'.format(
                i+1, args.loss_type, epoch_loss, epoch_loss_val, lr)
            tqdm.write(log)

            # Save the model with the best score
            if bestscore_train > epoch_loss:
                bestscore_train = epoch_loss
                bestscore_epoch_train = i+1
                savedata.save_model(model, 'best_model_train.pt')
                tqdm.write('\tBest train score: {:.4f}, Model saved'.format(bestscore_train))
            if bestscore_val > epoch_loss_val:
                bestscore_val = epoch_loss_val
                bestscore_epoch_val = i+1
                savedata.save_model(model, 'best_model_val.pt')
                tqdm.write('\tBest val score: {:.4f}, Model saved'.format(bestscore_val))

            # Save epoch loss
            savedata.write_loss(save_loss)

            pbar.update(1)

        # Save training time
        training_time = time.time() - start_time
        savedata.keep_records(training_time, bestscore_epoch_train, bestscore_epoch_val)


if __name__ == '__main__':
    start_time = time.time()
    args = option.parser().parse_args()
    main(args, start_time, set_loss, lr_scheduler)
    print("\n\nTraining time: %4.4fs\n" % (time.time() - start_time))
