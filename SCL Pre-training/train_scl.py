import torch
import argparse
import os
import random
import torch.optim as optim
import time
import numpy as np
import shutil

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nt_xent_sup_sparse import NT_Xent_Sup
from dataset import Kadis
from model5 import CModel



def lr_scheduler(optimizer, epoch, lr_decay_epoch=8):
    if epoch % lr_decay_epoch == 0:
        decay_rate = 0.9 ** (epoch // lr_decay_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer


def train_one(dataloader_train, model, loss_fn, 
              optimizer, scheduler, device):
    loss_epoch = []
    loop = tqdm(dataloader_train)
    for batch_idx, (dist, ref, label_dist) in enumerate(loop):
        dist = dist.to(device).float()
        ref = ref.to(device).float()

        label_ref = torch.zeros(label_dist.shape[1])
        label_ref[0] = 1
        label_ref = label_ref.repeat(label_dist.shape[0], 1)
        label = torch.cat([label_dist, label_ref], dim=0)

        z1, z2, _ = model(dist, ref)

        loss = loss_fn(z1, z2, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_epoch.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return sum(loss_epoch) / len(loss_epoch)


def main(args):
    root_dir = os.getcwd()
    root_dir = os.path.join(root_dir, 'dataset_iqa', 'kadis700k')

    # fix the seed if needed for reproducibility
    if args.seed == 0:
        pass
    else:
        print('SEED = {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # mkdir
    if not os.path.exists(args.sv_path):
        os.mkdir(args.sv_path)

    if not os.path.exists(args.tb_path):
        os.mkdir(args.tb_path)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)


    # build loader
    dset = Kadis(root_dir)
    dataloader_train = DataLoader(dset, batch_size=args.bsize, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = CModel(out_dim=args.out_dim).to(device)
    loss_fn = NT_Xent_Sup(args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tm)
#    scheduler = None

    writer = SummaryWriter(args.tb_path)

    start_epoch = -1
    # resume
    if args.resume:
        ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f'=> Loaded checkpoint from Epoch: {start_epoch}')
    
    
    loss_min = 100
    for epoch in range(start_epoch+1, args.epoch):
#        optimizer = lr_scheduler(optimizer, epoch)
        loss_epoch = train_one(
            dataloader_train, model, loss_fn, 
            optimizer, scheduler, device
        )
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch [{epoch}/{args.epoch}] |\t Loss: {loss_epoch:.5f} | LR: {cur_lr}')
        writer.add_scalar('Train_loss', loss_epoch, epoch)

        model.eval()

        ckpt = {
            'epoch': epoch,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if loss_epoch < loss_min:
            loss_min = loss_epoch
            torch.save(ckpt, os.path.join(args.model_path, f'{args.model_name}.pt'))
            model.train()
            print('=> Save checkpoint')


def clear_file(args):
    if os.path.exists(args.tb_path):
        shutil.rmtree(args.tb_path)
        print('=> Delete tensorboard file')

#    if os.path.exists(args.model_path):
#        shutil.rmtree(args.model_path)
#        print('=> Delete model file')

    print('Clear complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300, help='epoch')
    parser.add_argument('--dset', type=str, default='kadid10k', help='dataset')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning_rate')
    parser.add_argument('--bsize', type=int, default=110, help='batch_size')
    parser.add_argument('--psize', type=int, default=224, help='patch_size')
    parser.add_argument('--pnum', type=int, default=8, help='patch_num')
    parser.add_argument('--temperature', type=int, default=0.1, help='temperature')    
    parser.add_argument('--out_dim', type=int, default=128, help='out_dim')  
    parser.add_argument('--seed', type=int, default=2022, help='seed')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--tm', type=int, default=50, help='T_max')
    parser.add_argument('--sv_path', type=str, default='sav', help='save_path')
    parser.add_argument('--tb_path', type=str, default='sav/tb', help='tensorboard_path')
    parser.add_argument('--model_path', type=str, default='sav/model', help='save_model_path')
    parser.add_argument('--model_name', type=str, default='iqa-97-msb', help='model name')
    parser.add_argument('--resume', action='store_true', default=False)

    args = parser.parse_args()

    clear_file(args)
    main(args)
