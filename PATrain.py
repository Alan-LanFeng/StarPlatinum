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
    train_dataset = WaymoDataset(dataset_cfg, 'testing')
    print('len:', len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))
    # ================================evaluation Method==========================================
    evaluator = WODEvaluator(cfg, device, gpu_num)
    # =================================== INIT Model ============================================================
    vanilla_model = load_model_class(cfg['vanilla_model_name'])
    model_cfg = cfg['model_cfg']
    vanilla_model = vanilla_model(model_cfg, device)
    oracle_model = load_model_class(cfg['oracle_model_name'])
    oracle_model = oracle_model(model_cfg, device)

    if not args.local:
        vanilla_model = torch.nn.DataParallel(vanilla_model, list(range(gpu_num)))
        vanilla_model = vanilla_model.to(device)
        oracle_model = torch.nn.DataParallel(oracle_model, list(range(gpu_num)))
        oracle_model = oracle_model.to(device)

    if args.resume:
        resume_model_name = os.path.join(
            'saved_models', '{}.pt'.format(cfg['vanilla_ckpt']))
        vanilla_model = load_checkpoint(resume_model_name, vanilla_model, None, args.local)
        print('Successful Resume model {}'.format(resume_model_name))
        resume_model_name = os.path.join(
            'saved_models', '{}.pt'.format(cfg['oracle_ckpt']))
        oracle_model = load_checkpoint(resume_model_name, oracle_model, None, args.local)
        print('Successful Resume model {}'.format(resume_model_name))

    print('Finished Initializing in {:.3f}s!!!'.format(time.time() - start_time))
    # *====================================Training loop=======================================================
    print("Initial at {}:".format(time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    cnt = 0
    vanilla_model.eval()
    oracle_model.eval()
    progress_bar = tqdm(train_dataloader)
    for j, data in enumerate(progress_bar):
        for key in data.keys():
            if isinstance(data[key], torch.DoubleTensor):
                data[key] = data[key].float()
            if isinstance(data[key], torch.Tensor) and not args.local:
                data[key] = data[key].to('cuda:0')

        ego_index = 0
        coord, score, new_data = vanilla_model(data)
        coord = coord[:, ego_index]
        misc = new_data['misc']
        yaw = new_data['misc'][..., ego_index, 10, 4]
        s, c = torch.sin(yaw).view(*yaw.shape, 1, 1), torch.cos(yaw).view(*yaw.shape, 1, 1)
        coord[..., 0], coord[..., 1] = c * coord[..., 0] - s * coord[..., 1], \
                                       s * coord[..., 0] + c * coord[..., 1]
        centroid = new_data['centroid'][:, ego_index]
        centroid = centroid.view(*yaw.shape, 1, 1, 2)
        coord = coord.cumsum(-2) + centroid

        for k in range(cfg['model_cfg']['prop_num']):
            ego_future_path = coord[:, k]
            data['misc'][:, 0, 11:, :2] = ego_future_path
            data['misc'][:, 0, 11:, -2].fill_(1)

            ora_coord, ora_score, ora_new_data = oracle_model(data)