
import os
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.waymo_dataset import WaymoDataset
from utils.evaluator import WODEvaluator
from utils.utilities import load_model_class, load_checkpoint, save_checkpoint
from l5kit.configs import load_config_data
from utils.criterion import Loss
import matplotlib.pyplot as plt

cx, cy = 300, 300

dx = [-1, 0, 1]
dy = [-1, 0, 1]


def draw_line(bucket,x0, x1, y0, y1, cx=300, cy=300, color=0.3, verbose=False):
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    leng = int(((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5)
    leng = max(1, leng)
    if verbose:
        print(leng)
    for i in range(leng):
        mx = x0 + (x1 - x0) / leng * i
        my = y0 + (y1 - y0) / leng * i
        for xx in range(3):
            for yy in range(3):
                try:
                    bucket[int(mx) + dx[xx] + cx, int(my) + dy[yy] + cy] += color
                except:
                    import pdb;
                    pdb.set_trace()
    return bucket


def draw(data, new_data, pred, score, prefix):
    wash = lambda x: x.detach().cpu()
    for k in data.keys():
        try:
            data[k] = wash(data[k])
        except:
            pass
    for k in new_data.keys():
        new_data[k] = wash(new_data[k])
    show = new_data['tracks_to_predict']
    tmp_pred = pred.detach().cpu().numpy()
    pred = wash(pred)
    score = wash(score)
    centroid = new_data['centroid']

    gt = new_data['gt']
    # b,n,t,c = gt.shape

    # target_gt = data['groundtruth']
    # centroid = torch.cat([torch.zeros([b,1,2]), nb_centroid], dim=1)
    # gt = torch.cat([target_gt.view(b,t,2).unsqueeze(1), nb_gt[...,:2]], dim=1)
    # gt = gt.unsqueeze(2)

    # gt_ava = gt[...,2]
    gt_ava = new_data['gt_mask']
    # gt_ava = torch.cat([data['groundtruth_availabilities'].view(-1, 1, 80), gt_ava], axis=1)
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

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')
    for i in range(gt.shape[0]):
        lane = data['lane_vector']
        l = lane[i]
        adj_index = data['adj_index'][i][data['tracks_to_predict'][i]]
        adj_index = torch.unique(adj_index).unsqueeze(1).unsqueeze(1).repeat(1, 9, 11)
        l = torch.gather(l, 0, adj_index)
        n_line, n_point, n_channel = l.shape

        for j in range(n_line):
            for k in range(n_point):
                x0, y0, x1, y1, *one_hot = l[j, k]
                ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.3)

        p, n = 0, 0
        point = []
        for car in range(gt.shape[1]):
            if not show[i, car]:
                continue

            if cls[i, car]:
                p += 1
            else:
                n += 1

            point.append(ax.scatter(centroid[i, car, 0], centroid[i, car, 1], s=3))
            cur_yaw = yaw[i, car]
            s, c = np.sin(cur_yaw), np.cos(cur_yaw)
            for j in range(6):
                cl = pred[i, car, j].cumsum(-2)[:gt_ava[i, car], :]
                cl[:, 0], cl[:, 1] = c * cl[:, 0] - s * cl[:, 1], s * cl[:, 0] + c * cl[:, 1]
                cl = cl + centroid[i, car].unsqueeze(0)
                ax.plot(cl[:, 0], cl[:, 1], c=C[j], linewidth=0.6)

            gtc = gt[i, car].cumsum(-2)[:gt_ava[i, car], :]
            gtc[:, 0], gtc[:, 1] = c * gtc[:, 0] - s * gtc[:, 1], s * gtc[:, 0] + c * gtc[:, 1]
            gtc = gtc + centroid[i, car].unsqueeze(0)
            ax.plot(gtc[:, 0], gtc[:, 1], c=(1, 0, 0), linestyle='-' if cls[i, car] else '--', linewidth=0.3)

        global po, ne
        if p > 0:
            po += 1
        if n > 0:
            ne += 1

        figname = f'p-{p}-n-{n}' + '.png'
        while os.path.isfile(os.path.join('./vis', figname)):
            figname = '=' + figname

        plt.autoscale()
        fig.savefig(f'./vis/{figname}', dpi=300)
        for l in ax.get_lines():
            l.remove()
        for p in point:
            p.remove()
        if po >= P and ne >= N:
            return 1
    return 0

if __name__ == "__main__":
    # =================argument from systems====================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--cfg', type=str, default='0')
    parser.add_argument('--model-name', type=str, default='default_model')
    parser.add_argument('-P', '--positive', type=int, default=10)
    parser.add_argument('-N', '--negative', type=int, default=10)

    args = parser.parse_args()
    global P, N
    P = args.positive
    N = args.negative
    # ===============Choose-Data================================================================================
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

    # ================================== INIT DATASET ==========================================================
    start_time = time.time()

    dataset_cfg = cfg['dataset_cfg']
    train_dataset = WaymoDataset(dataset_cfg, 'validation_interactive')
    print('len:', len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))

    val_dataset = WaymoDataset(dataset_cfg, 'validation')
    val_loader = DataLoader(val_dataset,shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                            num_workers=dataset_cfg['num_workers'] * (not args.local))

    # =================================== INIT Model ============================================================
    model = load_model_class(cfg['model_name'])
    model_cfg = cfg['model_cfg']
    model = model(model_cfg)

    train_cfg = cfg['train_cfg']
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_decay_per_epoch'],
                                          gamma=train_cfg['decay_rate'])
    model = torch.nn.DataParallel(model, list(range(gpu_num)))
    if not args.local:
        #model = torch.nn.DataParallel(model, list(range(gpu_num)))
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
    global po, ne
    po, ne = 0, 0
    with torch.no_grad():
        model.eval()
        progress_bar = tqdm(train_dataloader)
        cnt = 0
        for j, data in enumerate(progress_bar):
            for key in data.keys():
                if isinstance(data[key], torch.DoubleTensor):
                    data[key] = data[key].float()
                if isinstance(data[key], torch.Tensor) and not args.local:
                    data[key] = data[key].to('cuda:0')
            coord, score, new_data = model(data)

            if draw(data, new_data, coord, score, j):
                break
