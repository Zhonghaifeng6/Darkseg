import argparse
import logging
import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.segformer import DarkSeg
from utils.dataset import BasicDataset
from utils.config import NetConfig
from utils.loss import LovaszLossSoftmax, LovaszLossHinge
from utils. loss import dice_coeff,MIOULoss,LovaszLossSoftmax_mutul
import os
import cv2


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
criterion_L2 = nn.MSELoss()
cfg = NetConfig()


def train_net(net, cfg):
    dataset = BasicDataset(cfg.images_dir, cfg.masks_dir, cfg.dark_dir, cfg.scale)

    val_percent = cfg.validation / 100
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{cfg.lr}_BS_{cfg.batch_size}_SCALE_{cfg.scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {cfg.epochs}
        Batch size:      {cfg.batch_size}
        Learning rate:   {cfg.lr}
        Optimizer:       {cfg.optimizer}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {cfg.save_cp}
        Device:          {device.type}
        Images scaling:  {cfg.scale}
    ''')

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=cfg.lr)

    elif cfg.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)

    else:
        optimizer = optim.SGD(net.parameters(),lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay,nesterov=cfg.nesterov)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.lr_decay_milestones,
                                               gamma = cfg.lr_decay_gamma)

    if cfg.n_classes > 1:
        criterion = LovaszLossSoftmax()
    else:
        criterion = LovaszLossHinge()

    for epoch in range(cfg.epochs):

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='img') as pbar:
            for batch in train_loader:

                batch_imgs = batch['image'] ## infrared
                batch_masks = batch['mask'] ## label
                batch_dark = batch['dark']  ## low-light

                # Canny figure
                canny_imgs = []
                for img in batch_imgs:

                    img_np = img.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
                    img_np = (img_np * 255).astype(np.uint8)  # [0,1] -> [0,255]

                    # to gray
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    edges_tensor = torch.tensor(edges_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

                    canny_imgs.append(edges_tensor)
                batch_cannys = torch.stack(canny_imgs)

                assert batch_imgs.shape[1] == cfg.n_channels, \
                        f'Network has been defined with {cfg.n_channels} input channels, ' \
                        f'but loaded images have {batch_imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
                batch_dark = batch_dark.to(device=device, dtype=torch.float32)
                batch_cannys = batch_cannys.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
                batch_masks = batch_masks.to(device=device, dtype=mask_type)


                net.train()

                ## training of the teacher model
                teacher_masks, _ = net(batch_masks)

                with torch.no_grad():
                    _, mid_teacher = net(batch_masks)

                ## training of the student model
                student_masks, mid_student = net(batch_dark)

                loss_s1 = criterion_L2(mid_student[0], mid_teacher[0]) / 32
                loss_s2 = criterion_L2(mid_student[1], mid_teacher[1]) / 64
                loss_s3 = criterion_L2(mid_student[2], mid_teacher[2]) / 160
                loss_s4 = criterion_L2(mid_student[3], mid_teacher[3]) / 256

                loss_in = criterion(teacher_masks, batch_cannys)

                loss_sm = loss_s1 + loss_s2 + loss_s3 + loss_s4
                loss_be = criterion(teacher_masks, batch_masks)
                loss_low = loss_be + loss_sm

                loss_total = loss_low + loss_in


                epoch_loss += loss_total.item()
                writer.add_scalar('Loss/train', loss_total.item(), global_step)
                writer.add_scalar('model/lr', optimizer.param_groups[0]['lr'], global_step)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()


                pbar.set_postfix(ordered_dict={
                    "loss: ": loss_total.item(),
                })

                pbar.update(batch_imgs.shape[0])
                global_step += 1


                if global_step % (len(dataset) // (1 * cfg.batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val, cfg)
                    if cfg.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('CrossEntropy/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', batch_imgs, global_step)
                    if cfg.deepsupervision:
                            inference_masks = inference_masks[-1]
                    if cfg.n_classes == 1:
                        # writer.add_images('masks/true', batch_masks, global_step)
                        inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
                        writer.add_images('masks/inference',
                                          inference_mask,
                                          global_step)
                    else:
                        # writer.add_images('masks/true', batch_masks, global_step)
                        ids = inference_masks.shape[1]  # N x C x H x W
                        inference_masks = torch.chunk(inference_masks, ids, dim=1)
                        for idx in range(0, len(inference_masks)):
                            inference_mask = torch.sigmoid(inference_masks[idx]) > cfg.out_threshold
                            writer.add_images('masks/inference_'+str(idx),
                                              inference_mask,
                                              global_step)


        if cfg.save_cp:
            try:
                os.mkdir(cfg.checkpoints_dir)
                os.mkdir(cfg.checkpoints_dir1)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            if epoch % 5 == 0 :
                ckpt_name = 'epoch_' + str(epoch + 1) + '.pth'
                torch.save(net.state_dict(),
                           osp.join(cfg.checkpoints_dir, ckpt_name))
                # torch.save(net1.state_dict(),
                #            osp.join(cfg.checkpoints_dir1, ckpt_name))
    writer.close()


def eval_net(net, loader, device, n_val, cfg):
    """
    Evaluation without the densecrf with the dice coefficient

    """
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['dark']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # compute loss
            if cfg.deepsupervision:
                masks_preds,_ = net(imgs)
                loss = 0
                for masks_pred in masks_preds:
                    tot_cross_entropy = 0
                    for true_mask, pred in zip(true_masks, masks_pred):
                        pred = (pred > cfg.out_threshold).float()
                        if cfg.n_classes > 1:
                            sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        else:
                            sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        tot_cross_entropy += sub_cross_entropy
                    tot_cross_entropy = tot_cross_entropy / len(masks_preds)
                    tot += tot_cross_entropy
            else:
                masks_pred,_ = net(imgs)
                for true_mask, pred in zip(true_masks, masks_pred):
                    pred = (pred > cfg.out_threshold).float()
                    if cfg.n_classes > 1:
                        tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                    else:
                        tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()

            pbar.update(imgs.shape[0])

    return tot / n_val


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0')
    logging.info(f'Using device {device}')

    net = eval(cfg.model)(cfg)
    logging.info(f'Network:\n'
                 f'\t{cfg.model} model\n'
                 f'\t{cfg.n_channels} input channels\n'
                 f'\t{cfg.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if cfg.bilinear else "Dilated conv"} upscaling')

    if cfg.load:
        net.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')

    net.to(device=device)

    try:
        train_net(net=net, net1=net ,cfg=cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
