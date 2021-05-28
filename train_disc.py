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
from torch.autograd import Variable
from models.killer_queen import killer_queen

# ==============================Main=======================================
if __name__ == "__main__":
    # =================argument from systems====================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--cfg', type=str, default='0')
    parser.add_argument('--model-name', type=str, default='default_model')
    args = parser.parse_args()

    cfg = load_config_data(f"./config/{args.cfg}.yaml")
    device = 'cpu' if args.local else 'cuda'

    # print(cfg)
    if device == 'cpu':
        gpu_num = 1
        print('device: CPU')
    else:
        gpu_num = torch.cuda.device_count()
        print("gpu number:{}".format(gpu_num))
        print("gpu available:", torch.cuda.is_available())

    writer = SummaryWriter()
    # ================================== INIT DATASET ==========================================================
    start_time = time.time()

    dataset_cfg = cfg['dataset_cfg']
    dataset_cfg['cache'] = True
    train_dataset = WaymoDataset(dataset_cfg, 'training')
    print('len:', len(train_dataset))


    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))

    if cfg['track'] == 'interaction':
        val_dataset = WaymoDataset(dataset_cfg, 'validation')
    else:
        val_dataset = WaymoDataset(dataset_cfg, 'validation')
    val_loader = DataLoader(val_dataset,shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))

    evaluator = WODEvaluator(cfg, device,val_loader)
    # =================================== INIT Model ============================================================
    model = load_model_class(cfg['model_name'])
    model_cfg = cfg['model_cfg']
    model = model(model_cfg)


    train_cfg = cfg['train_cfg']
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)



    disc = killer_queen(model_cfg)
    optimizer_D = optim.AdamW(disc.parameters(), lr=train_cfg['lr'],
                              betas=(0.9, 0.999), eps=1e-09,
                              weight_decay=train_cfg['weight_decay'],
                              amsgrad=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_decay_per_epoch'],
                                          gamma=train_cfg['decay_rate'])

    if not args.local:
        model = torch.nn.DataParallel(model, list(range(gpu_num)))
        model = model.to(device)
        disc = torch.nn.DataParallel(disc, list(range(gpu_num)))
        disc = disc.to(device)

    if args.resume:
        resume_model_name = os.path.join(
            'saved_models', '{}.pt'.format(args.model_name))
        model = load_checkpoint(resume_model_name, model, optimizer, args.local)
        print('Successful Resume model {}'.format(resume_model_name))
    loss_cfg = cfg['loss_cfg']
    criterion = Loss(loss_cfg, device)

    print('Finished Initializing in {:.3f}s!!!'.format(time.time() - start_time))

    # *====================================Training loop=======================================================
    print("Initial at {}:".format(time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    cnt = 0
    max_epoch = 500
    for epoch in range(max_epoch):
        model.train()
        print(f'{epoch + 1}/{max_epoch} start at ' +
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        progress_bar = tqdm(train_dataloader)
        for j, data in enumerate(progress_bar):
            # Checking data preprocess
            for key in data.keys():
                if isinstance(data[key], torch.DoubleTensor):
                    data[key] = data[key].float()
                if isinstance(data[key], torch.Tensor) and not args.local:
                    data[key] = data[key].to('cuda:0')

            optimizer.zero_grad()
            output = model(data)
            #L2 loss
            loss, losses, miss_rate,index = criterion(output,cfg['track'])

            Tensor = torch.cuda.FloatTensor if not args.local else torch.FloatTensor
            valid = Variable(Tensor(*output[0].shape[:3], 1).fill_(1.0), requires_grad=False)
            adversarial_loss = torch.nn.BCELoss(reduction='none')
            loss_mask = output[2]['tracks_to_predict']

            conf = disc(output[3],loss_mask)
            loss_g = torch.mean(adversarial_loss(conf, valid).squeeze(-1),-1)
            loss_g = loss_g*loss_mask
            loss_g = loss_g.sum()/max(loss_mask.sum(),1)
            loss = loss_g + loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg['max_norm_gradient'])
            optimizer.step()

            #==============train disc====================
            optimizer_D.zero_grad()
            valid = Variable(Tensor(*output[0].shape[:2], 1, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(*output[0].shape[:2], 1, 1).fill_(0.0), requires_grad=False)
            input = output[3]
            for k,v in input.items():
                input[k] = v.detach()
            input['pred_mask'] = input['gt_mask']

            index = index.unsqueeze(-1).repeat(1,2)

            gather_traj = index.view(*index.shape,1,1,1).repeat(1,1,1,*input['traj'].shape[-2:])
            input['traj'] = torch.gather(input['traj'],2,gather_traj)
            fake_loss = adversarial_loss(disc(input,loss_mask), fake).squeeze(-1).squeeze(-1)

            input['traj'] = input['gt_traj']
            real_loss = adversarial_loss(disc(input,loss_mask), valid).squeeze(-1).squeeze(-1)

            real_loss = (real_loss*loss_mask).sum()/loss_mask.sum()
            fake_loss = (fake_loss*loss_mask).sum()/loss_mask.sum()
            loss_d = (real_loss + fake_loss) / 2

            loss_d.backward()
            optimizer_D.step()

            losses_text = ''
            for loss_name in losses:
                losses_text += loss_name + ':{:.3f} '.format(losses[loss_name])
            progress_bar.set_description(desc='{} total-MR:{:.1f}% loss_g:{:.3f} loss_d:{:.3f}'.format(losses_text, miss_rate * 100,loss_g,loss_d))

            log_dict = {"loss/totalloss": loss.detach(), "loss/reg": losses['reg_loss'], "loss/cls": losses['cls_loss'],
                        "loss/loss_g": loss_g.detach(),
                        "loss/loss_d": loss_d.detach(),
                        "loss/crash": losses['crash_loss'],
                        'MR': miss_rate}

            for k, v in log_dict.items():
                writer.add_scalar(k, v, cnt)
            cnt += 1

        scheduler.step()

        eval_dict = evaluator.evaluate(model)
        for k, v in eval_dict.items():
            writer.add_scalar(k, v, cnt)

        # save after every epoch
        if not os.path.exists('./saved_models/'):
            os.mkdir('./saved_models/')
        model_save_name = os.path.join(
            'saved_models', '{}_{}.pt'.format(args.model_name, epoch + 1))
        save_checkpoint(model_save_name, model, optimizer)
        disc_save_name = os.path.join(
            'saved_models', 'd_{}_{}.pt'.format(args.model_name, epoch + 1))
        save_checkpoint(disc_save_name, disc, optimizer_D)


        print('Epoch{} finished!!!!'.format(epoch + 1))

    print('Trainning Process Finished at {}!!'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
