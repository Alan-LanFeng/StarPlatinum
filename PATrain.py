import os
import time
import argparse
import torch
import numpy as np
import math

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.waymo_dataset import WaymoDataset
from utils.evaluator import WODEvaluator
from utils.utilities import load_model_class, load_checkpoint, save_checkpoint
from l5kit.configs import load_config_data
from utils.criterion import Loss


class Canvas:
    def __init__(self, L, R, D, U):
        self.L = L
        self.R = R
        self.D = D
        self.U = U
        self.canvas = np.zeros([int(R - L)+1, int(U - D)+1])

    def _draw_dot(self, x, y, value):
        cx = int(x - self.L)
        cy = int(y - self.D)
        self.canvas[cx, cy] += value

    def _draw_line(self, x0, y0, x1, y1, value_f, k=2):
        length = ((x0-x1)**2+(y0-y1)**2)**0.5
        interval_num = max(int(length * k), 1)
        dx = (x1 - x0) / interval_num
        dy = (y1 - y0) / interval_num
        for i in range(interval_num):
            _x = x0 + dx * i
            _y = y0 + dy * i
            self._draw_dot(_x, _y, value_f(_x, _y))

    def draw(self, xist0, yist0, xist1, yist1, value_f):
        # x: list of x-coordinates
        # y: list of y-coordinates
        # value_f: lambda function
        for x0, y0, x1, y1 in zip(xist0, yist0, xist1, yist1):
            self._draw_line(x0, y0, x1, y1, value_f)

    def to_image(self, name):
        img = Image.fromarray(self.canvas * 256).convert('L')
        img.save(f'./canvas/{name}.png')

def loss_function(data, idx, coord, score, new_data, ora_coord, ora_score, ora_new_data):
    lane = data['lane_vector']  # batch_size, max_lane_num, 9, 6
    lane_flat_x = lane[..., (0,2)].view(lane.shape[0], -1)
    lane_flat_y = lane[..., (1,3)].view(lane.shape[0], -1)
    L, R = lane_flat_x.min(-1).values, lane_flat_x.max(-1).values
    D, U = lane_flat_y.min(-1).values, lane_flat_y.max(-1).values
    batch_size = coord.shape[0]

    yaw = ora_new_data['misc'][..., 10, 4]
    yaw = yaw.view(*yaw.shape, 1, 1)
    s, c = torch.sin(yaw), torch.cos(yaw)
    center = ora_new_data['centroid']
    center = center.view(*center.shape[:2], 1, 1, 2)
    ora_coord = ora_coord.cumsum(-2)
    ora_coord[..., 0], ora_coord[..., 1] = c * ora_coord[..., 0] - s * ora_coord[..., 1], \
                                           s * ora_coord[..., 0] + c * ora_coord[..., 1]
    ora_coord += center

    for i in range(batch_size):
        canvas = Canvas(L[i], R[i], D[i], U[i])
        # 1. build cost map with LANE INFO
        cur_lane = lane[i]
        cur_lane = cur_lane[cur_lane[..., 4] == 15]
        cur_lane = cur_lane[cur_lane[..., 5] == 1][..., :4].T.tolist()
        canvas.draw(cur_lane[0], cur_lane[1], cur_lane[2], cur_lane[3], lambda x, y: 0.2)

        # car iteration
        for j in range(ora_coord.shape[1]):
            # proposal iteration
            for k in range(ora_coord.shape[2]):
                # build cost map with ora-COORD
                ora_pred = ora_coord[i, j, k]
                ora_pred = ora_pred.T.tolist()
                canvas.draw(ora_pred[0][:-1], ora_pred[1][:-1], ora_pred[0][1:], ora_pred[1][1:], lambda x, y: 0.5)
        canvas.to_image(i)

        # calculate pro-active proposal loss


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
            loss_function(data, k, coord, score, new_data, ora_coord, ora_score, ora_new_data)
            assert 1 == 0
