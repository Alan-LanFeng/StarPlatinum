import os
import time
import argparse
import torch
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.waymo_dataset import WaymoDataset
from utils.evaluator import WODEvaluator
from utils.utilities import load_model_class, load_checkpoint, save_checkpoint
from l5kit.configs import load_config_data
from utils.criterion import Loss
import matplotlib.pyplot as plt

class Draw:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.axis('equal')

    def draw(self, data, new_data, ego, pred, score, prefix, batch_id):
        wash = lambda x: x.detach().cpu()
        for k in data.keys():
            try:
                data[k] = wash(data[k])
            except:
                pass
        for k in new_data.keys():
            new_data[k] = wash(new_data[k])
        ego = wash(ego)
        show = new_data['tracks_to_predict']
        tmp_pred = pred.detach().cpu().numpy()
        pred = wash(pred)
        score = wash(score)
        centroid = new_data['centroid']

        gt = new_data['gt']
        gt_ava = new_data['gt_mask']
        gt_ava = gt_ava.cumsum(-1) == torch.ones_like(gt_ava).cumsum(-1)
        gt_ava = gt_ava.sum(-1) - 1

        dis = gt.cumsum(-2).unsqueeze(2) - pred.cumsum(-2)
        gt_ava[gt_ava < 0] = 0
        dis = torch.gather(dis, dim=-2, index=gt_ava.view(*gt_ava.shape, 1, 1, 1).repeat(1, 1, *dis.shape[2:]))
        dis = np.linalg.norm(dis[..., -1, :], ord=2, axis=-1) < 8  # [batchsize, carnum, K]
        cls = dis.sum(-1) > 0  # [batchsize, carnum]
        # yaw = torch.cat((data['ego_groundtruth_misc'][:,10,4].unsqueeze(1), data['nbrs_groundtruth_misc'][:,:,10,4]), dim=1)
        yaw = new_data['misc'][:, :, 10, 4]

        gt = gt.squeeze(2)
        C = [
            (0, 0.3, 1),
            (0, 0.4, 0.9),
            (0, 0.5, 0.8),
            (0, 0.6, 0.7),
            (0, 0.7, 0.6),
            (0, 0.8, 0.5)
        ]

        i = batch_id
        lane = data['lane_vector']
        l = lane[i]
        adj_index = data['adj_index'][i][data['tracks_to_predict'][i]]
        adj_index = torch.unique(adj_index).unsqueeze(1).unsqueeze(1).repeat(1, 9, 11)
        l = torch.gather(l, 0, adj_index)
        n_line, n_point, n_channel = l.shape

        for j in range(n_line):
            for k in range(n_point):
                x0, y0, x1, y1, *one_hot = l[j, k]
                self.ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.3)

        p, n = 0, 0
        point = []
        self.ax.plot(ego[:, 0], ego[:, 1], 'orange', linewidth=1)
        is_first = True
        for car in range(gt.shape[1]):
            if not show[i, car]:
                continue
            if show[i, car] and is_first:
                is_first = False
                continue

            if cls[i, car]:
                p += 1
            else:
                n += 1

            point.append(self.ax.scatter(centroid[i, car, 0], centroid[i, car, 1], s=3))
            cur_yaw = yaw[i, car]
            s, c = np.sin(cur_yaw), np.cos(cur_yaw)

            for j in range(6):
                cl = pred[i, car, j].cumsum(-2)[:gt_ava[i, car], :]
                cl[:, 0], cl[:, 1] = c * cl[:, 0] - s * cl[:, 1], s * cl[:, 0] + c * cl[:, 1]
                cl = cl + centroid[i, car].unsqueeze(0)
                self.ax.plot(cl[:, 0], cl[:, 1], c=C[j], linewidth=0.6)

            gtc = gt[i, car].cumsum(-2)[:gt_ava[i, car], :]
            gtc[:, 0], gtc[:, 1] = c * gtc[:, 0] - s * gtc[:, 1], s * gtc[:, 0] + c * gtc[:, 1]
            gtc = gtc + centroid[i, car].unsqueeze(0)
            self.ax.plot(gtc[:, 0], gtc[:, 1], c=(1, 0, 0), linestyle='-' if cls[i, car] else '--', linewidth=0.3)

        figname = 'p' + prefix + '.png'
        while os.path.isfile(os.path.join('./vis', figname)):
            figname = '=' + figname

        plt.autoscale()
        self.fig.savefig(f'./vis/{figname}', dpi=300)
        for l in self.ax.get_lines():
            l.remove()
        for p in point:
            p.remove()

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
    train_dataset = WaymoDataset(dataset_cfg, 'validation')
    print('len:', len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))
    # =================================== INIT Model ============================================================
    vanilla_model = load_model_class(cfg['vanilla_model_name'])
    vanilla_model_cfg = cfg['vanilla_model_cfg']
    oracle_model_cfg = cfg['oracle_model_cfg']
    vanilla_model = vanilla_model(vanilla_model_cfg)
    oracle_model = load_model_class(cfg['oracle_model_name'])
    oracle_model = oracle_model(oracle_model_cfg)

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
    loss_cmr_total = 0
    pred_cmr_total = 0

    a,b,cc,d,e,g,h = [], [], [], [], [], [], []
    xxx = 0
    for j, data in enumerate(progress_bar):
        for key in data.keys():
            if isinstance(data[key], torch.DoubleTensor):
                data[key] = data[key].float()
            if isinstance(data[key], torch.Tensor) and not args.local:
                data[key] = data[key].to('cuda:0')

        coord, score, new_data = vanilla_model(data)

        i = 0
        ego_index = torch.where(new_data['tracks_to_predict'][i] == True)[0][0]
        coord = coord[:, ego_index]
        yaw = new_data['misc'][..., ego_index, 10, 4]
        s, c = torch.sin(yaw).view(*yaw.shape, 1, 1), torch.cos(yaw).view(*yaw.shape, 1, 1)
        coord[..., 0], coord[..., 1] = c * coord[..., 0] - s * coord[..., 1], \
                                       s * coord[..., 0] + c * coord[..., 1]
        centroid = new_data['centroid'][:, ego_index]
        centroid = centroid.view(*yaw.shape, 1, 1, 2)
        coord = coord.cumsum(-2) + centroid

        i = 0
        draw = Draw()
        import copy
        for k in range(cfg['vanilla_model_cfg']['prop_num']):
            cdata = copy.deepcopy(data)
            ego_future_path = coord[i, k]
            idx = torch.where(cdata['tracks_to_predict'][i] == True)[0][0]
            cdata['misc'][i, idx, 11:, :2] = ego_future_path.detach().clone()
            cdata['misc'][i, idx, 11:, -2].fill_(1)

            ora_coord, ora_score, ora_new_data = oracle_model(cdata)
            draw.draw(data, ora_new_data, ego_future_path, ora_coord, ora_score, f'{j}-{k}', i)

