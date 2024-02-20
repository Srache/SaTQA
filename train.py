import torch
import torch.nn as nn
import argparse
import os
import logging
import random
import numpy as np
import torch.optim as optim
import shutil
import json

from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import DataLoader
from tqdm import tqdm
from utils.util import calc_coefficient
from torchvision.models import ResNet50_Weights
from utils.model_q import NIQA
from utils.model_scl import CModel


def train_fn(args):
    root_dir = os.getcwd()
    live_path = os.path.join(root_dir, 'dataset_iqa', 'live')
    csiq_path = os.path.join(root_dir, 'dataset_iqa', 'csiq')
    livec_path = os.path.join(root_dir, 'dataset_iqa', 'livec')
    kadid10k_path = os.path.join(root_dir, 'dataset_iqa', 'kadid10k')
    koniq_path = os.path.join(root_dir, 'dataset_iqa', 'koniq')
    tid2013_path = os.path.join(root_dir, 'dataset_iqa', 'tid2013')
    fblive_path = os.path.join(root_dir, 'dataset_iqa', 'fblive')

    folder_path = {
        'live': live_path,
        'csiq': csiq_path,
        'tid2013': tid2013_path,
        'kadid10k': kadid10k_path,
        'livec': livec_path,
        'koniq': koniq_path,
        'fblive': fblive_path,
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        'fblive': list(range(0, 39810),)
    }

    print('Training and Testing on <{}> dataset'.format(args.dset.upper()))

    # SEED
    if args.seed == 0:
        pass
    else:
        print('SEED = {}'.format(args.seed))
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.sv_path):
        os.mkdir(args.sv_path)

    if not os.path.exists(args.tb_path):
        os.mkdir(args.tb_path)

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # log setting
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(args.log_path + f'/log_{args.dset}.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f'Dataset: {args.dset}')
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Batch_size: {args.bsize}')
    logger.info(f'Patch_size: {args.psize}')
    logger.info(f'Patch_num: {args.pnum}')
    logger.info(f'Seed: {args.seed}')
    logger.info(f'Weigh_decay: {args.wd}')
    logger.info(f'T_max: {args.tm}')

    total_num_images = img_num[args.dset]
    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]
    
    train_idx_path = f'{args.log_path}' + '/' + 'train_idx' +'_' + str(args.seed) + '.json'
    test_idx_path = f'{args.log_path}' + '/' + 'test_idx' +'_' + str(args.seed) + '.json'
    
    with open(train_idx_path, 'w') as f:
        json.dump(train_index, f)
        
    with open(test_idx_path, 'w') as f:
        json.dump(test_index, f)
    
    # build train and test loader
    dataloader_train = DataLoader(args.dset, folder_path[args.dset],
                                  train_index, args.psize, args.pnum,
                                  args.bsize, istrain=True).get_data()

    dataloader_test = DataLoader(args.dset, folder_path[args.dset],
                                 test_index, args.psize, args.tnum,
                                 istrain=False).get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = NIQA().to(device)
    loss_fn = nn.L1Loss()


    # encoder
    with torch.no_grad():
        cmodel = CModel().to(device)
        ckpt = torch.load(f'utils/{args.model_name_scl}.pt', map_location=device)
        cmodel.load_state_dict(ckpt['net'])
        cmodel.eval()

    for param in cmodel.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tm)

    writer = SummaryWriter(args.tb_path)
    
    start_epoch = -1
    # resume
    if args.resume:
        ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f'=> Loaded checkpoint from Epoch: {start_epoch}')
    
    best_srocc = 0.0
    best_plcc = 0.0
    for epoch in range(start_epoch+1, args.epoch):
        losses = []
        print(f'+====================+ Training Epoch: {epoch} +====================+')
        logger.info(f'Training on <{args.dset.upper()}>, EPOCH: {epoch}')
        loop = tqdm(dataloader_train)
        for batch_idx, (dist, rating) in enumerate(loop):
            batch_size = dist.shape[0]
            dist = dist.to(device).float()
            rating = rating.reshape(batch_size, -1).to(device).float()

            _, _, fea = cmodel(dist, dist)
            
            out = model(fea)
            loss = loss_fn(out, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss)
            loop.set_postfix(loss=loss.item())

        print(f'Loss: {sum(losses)/len(losses):.5f}')
        logger.info(f'Loss: {sum(losses)/len(losses):.5f}')
        writer.add_scalar('Train_loss', sum(losses)/len(losses), epoch)

        print(f'+====================+ Testing Epoch: {epoch} +====================+')
        sp, pl = calc_coefficient(dataloader_test, model, device, args.pnum, cmodel)
        print(f'SROCC: {sp:.4f}, PLCC: {pl:.4f}')
        logger.info(f'SROCC: {sp:.4f}, PLCC: {pl:.4f}')
        writer.add_scalar('SROCC', sp, epoch)
        writer.add_scalar('PLCC', pl, epoch)

        if sp > best_srocc:
            print('=> Save checkpoint')
            best_srocc = sp
            best_plcc = pl
            logger.info('=> Save checkpoint')

            model.eval()
            ckpt = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
    
            torch.save(ckpt, os.path.join(args.model_path, f'{args.model_name}.pth'))
            model.train()
            
        print(f'BEST SROCC: {best_srocc:.4f}, PLCC: {best_plcc:.4f}')


def clear_file(args):
    if os.path.exists(args.tb_path):
        shutil.rmtree(args.tb_path)
        print('====> Delete tensorboard file')

    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
        print('====> Delete log file')

    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)
        print('====> Delete model file')

    print('Clear complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=300, help='epoch')
    parser.add_argument('--dset', type=str, default='live', help='dataset')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning_rate')
    parser.add_argument('--bsize', type=int, default=8, help='batch_size')
    parser.add_argument('--psize', type=int, default=224, help='patch_size')
    parser.add_argument('--pnum', type=int, default=8, help='train patch_num')
    parser.add_argument('--tnum', type=int, default=8, help='test patch_num')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight_decay')
    parser.add_argument('--tm', type=int, default=50, help='T_max')
    parser.add_argument('--sv_path', type=str, default='sav', help='save_path')
    parser.add_argument('--tb_path', type=str, default='sav/tensorboard', help='tensorboard_path')
    parser.add_argument('--log_path', type=str, default='sav/log', help='log_path')
    parser.add_argument('--model_path', type=str, default='sav/model', help='save_model_path')

    parser.add_argument('--model_name_scl', type=str, default='iqa-97o', help='SCL model name')

    parser.add_argument('--model_name', type=str, default='niqa', help='model name')
    parser.add_argument('--resume', action='store_true', default=False)
    
    args = parser.parse_args()

    # clear_file(args)
    train_fn(args)

