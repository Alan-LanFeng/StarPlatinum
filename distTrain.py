import os
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.waymo_dataset import WaymoDataset
from utils.evaluator import WODEvaluator
from utils.utilities import load_model_class, load_checkpoint, save_checkpoint
from l5kit.configs import load_config_data
from utils.criterion import Loss
from utils.dist_helper import finalize
import pickle
from utils.dist_helper import DistributedSampler
import multiprocessing as mp
from utils.dist_helper import dist_init, DistModule

import linklink as link

if __name__ == "__main__":
    # =================preprocess====================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--cfg', type=str, default='0')
    parser.add_argument('--model-name', type=str, default='default_model')
    args = parser.parse_args()

    device = 'cuda'
    mp.set_start_method('fork', force=True)
    # linklink
    rank, world_size = dist_init()
    cfg = load_config_data(f"./config/{args.cfg}.yaml")
    writer = SummaryWriter()
    # =================buidl model===================================================
    model = load_model_class(cfg['model_name'])
    model_cfg = cfg['model_cfg']
    model = model(model_cfg)

    model = DistModule(model, sync=True)
    model.cuda()
    # ==================================set up dataloader==========================================================
    dataset_cfg = cfg['dataset_cfg']
    workers = dataset_cfg['num_workers']
    batch_size = dataset_cfg['batch_size']
    # build dataset
    train_dataset = WaymoDataset(dataset_cfg, 'training')
    # build sampler
    sampler = DistributedSampler(train_dataset)
    # build loader
    train_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=workers, sampler=sampler, shuffle=False, pin_memory=False)

    dataset_cfg = cfg['dataset_cfg']
    workers = dataset_cfg['num_workers']
    batch_size = dataset_cfg['batch_size']
    # build dataset
    val_dataset = WaymoDataset(dataset_cfg, 'validation')
    # build sampler
    sampler = DistributedSampler(val_dataset)
    # build loader
    val_loader = DataLoader(val_dataset,batch_size=batch_size, num_workers=workers, sampler=sampler, shuffle=False, pin_memory=False)

    # ===================================optimizer======================================================
    train_cfg = cfg['train_cfg']
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)

    # ===================================setup_lr_schedule======================================================
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_decay_per_epoch'],
                                          gamma=train_cfg['decay_rate'])

    # ================================evaluation Method==========================================
    evaluator = WODEvaluator(cfg, device,val_loader)

    # =================================== setup_criterion ============================================================
    loss_cfg = cfg['loss_cfg']
    criterion = Loss(loss_cfg, device)
    if rank == 0:
        writer = SummaryWriter()

    # start training
    for epoch in range(500):
        # train
        losses = {}  # dict of AverageMeter
        # switch to train mode
        model.train()
        # setup dataloader
        train_loader.sampler.set_epoch(epoch)

        end = time.time()
        for step, data in enumerate(train_loader):
            # update learning rate
            # lr = self.lr_scheduler.get_lr()
            scheduler.step()

            # forward
            tmp = time.time()
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and not args.local:
                    data[key] = data[key].cuda()

            outputs = model(data)
            # criterion
            loss, losses, miss_rate = criterion(outputs)
            loss = loss / world_size

            # backward
            optimizer.zero_grad()
            loss.backward()
            model.sync_gradients()
            optimizer.step()
            # update loss record

            for name, aloss in losses.items():
                loss_cpy = aloss.clone()  # .detach()
                link.allreduce(loss_cpy)
                losses['name'] = loss_cpy.item() / world_size
            link.allreduce(loss)
            link.allreduce(miss_rate)
            if rank == 0:
                log_dict = {"loss/totalloss": loss.detach(), "loss/reg": losses['reg_loss'], "loss/cls": losses['cls_loss'],
                            'MR': miss_rate}
            for k, v in log_dict.items():
                writer.add_scalar(k, v, step)

        if rank == 0:
            eval_dict = evaluator.evaluate(model)
            for k, v in eval_dict.items():
                writer.add_scalar(k, v, epoch)

        # only save in rank 0 process
        if rank != 0: continue

        if not os.path.exists('./saved_models/'):
            os.mkdir('./saved_models/')
        model_save_name = os.path.join(
            'saved_models', '{}_{}.pt'.format(args.model_name, epoch + 1))

        save_checkpoint(model_save_name, model, optimizer)
    # finalize link
    finalize()

