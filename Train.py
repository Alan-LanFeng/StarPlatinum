import os
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.waymo_dataset_v2 import WaymoDataset
from utils.evaluator import WODEvaluator
from utils.utilities import load_model_class, load_checkpoint, save_checkpoint
from l5kit.configs import load_config_data
from utils.criterion import Loss
import pickle
# =========================evaluation======================================
from torch.autograd import Variable

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

    # check if there's cache
    dataset_cfg = cfg['dataset_cfg']
    dir = dataset_cfg['dataset_dir']
    cache_root = dir[:dir.find('trans')]
    cache_path = os.path.join(cache_root, dataset_cfg['cache_name'])
    if not os.path.exists(cache_path):
        print('starting cache')
        dataset_cfg['cache'] = False
        periods = ['training', 'validation', 'testing', 'validation_interactive', 'testing_interactive']
        batch_size = dataset_cfg['batch_size']
        for period in periods:
            ds = WaymoDataset(dataset_cfg, period)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=dataset_cfg['num_workers'])
            progress_bar = tqdm(loader)
            cnt = 0
            for data in progress_bar:
                try:
                    for k, v in data.items():
                        data[k] = data[k].numpy()
                except:
                    pass
                path_name = os.path.join(cache_root, dataset_cfg['cache_name'], period)
                if not os.path.exists(path_name):
                    os.makedirs(path_name)
                for i in range(batch_size):
                    cache_file = os.path.join(path_name, f'{cnt}.pkl')

                    if not os.path.exists(cache_file):
                        os.mknod(cache_file)
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump({k: v[i] for k, v in data.items()}, f)
                    except:
                        pass
                    cnt += 1
    print("using existing cache")

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
    # ================================evaluation Method==========================================
    evaluator = WODEvaluator(cfg, device)
    # =================================== INIT Model ============================================================
    model = load_model_class(cfg['model_name'])
    model_cfg = cfg['model_cfg']
    model = model(model_cfg)

    train_cfg = cfg['train_cfg']
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_decay_per_epoch'],
                                          gamma=train_cfg['decay_rate'])

    if not args.local:
        model = torch.nn.DataParallel(model, list(range(gpu_num)))
        model = model.to(device)

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
            loss, losses, miss_rate = criterion(output)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg['max_norm_gradient'])
            optimizer.step()
            # * Display the results.


            losses_text = ''
            for loss_name in losses:
                losses_text += loss_name + ':{:.3f} '.format(losses[loss_name])
            progress_bar.set_description(desc='{} total-MR:{:.1f}% '.format(losses_text, miss_rate * 100))

            log_dict = {"loss/totalloss": loss.detach(), "loss/reg": losses['reg_loss'], "loss/cls": losses['cls_loss'],
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

        print('Epoch{} finished!!!!'.format(epoch + 1))

    print('Trainning Process Finished at {}!!'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
