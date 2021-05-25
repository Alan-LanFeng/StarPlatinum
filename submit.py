from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos import motion_submission_pb2
import os

import time
import argparse
import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.waymo_dataset import WaymoDataset
from l5kit.configs import load_config_data

from utils.utilities import (load_checkpoint, save_checkpoint, load_model_class,
                                 vis_argoverse, set_model_grad, fix_parameter_except)


def rotate(x, theta):
    s, c = np.sin(theta), np.cos(theta)
    x[..., 0], x[..., 1] = c * x[..., 0] - s * x[..., 1], \
                           s * x[..., 0] + c * x[..., 1]
    return x


class Submit:
    def __init__(self):
        self.submission = motion_submission_pb2.MotionChallengeSubmission()

        # meta info
        self.submission.submission_type = 1
        self.submission.account_name = 'zqh10241024@gmail.com'
        self.submission.unique_method_name = 'mmTrans'

        self.cnt = 0
        self.last_cnt = 0

    def fill(self, output, data, new_data):
        # The set of scenario predictions to evaluate.
        # One entry should exist for every record in the val/test set.

        wash = lambda x: x.detach().cpu().numpy()
        for k in data.keys():
            try:
                data[k] = wash(data[k])
            except:
                pass
        for k in output.keys():
            output[k] = wash(output[k])
        for k in new_data.keys():
            new_data[k] = wash(new_data[k])

        coord = output['pred_coords']  # example: 32, 8, 6, 80, 2
        coord = coord.cumsum(-2)
        logit = output['pred_logits']  # example: 32, 8, 6
        idx = np.argsort(logit, -1)[...,::-1]
        centroid = new_data['centroid']
        batch_size, car_num, K = coord.shape[:3]
        for i in range(batch_size):
            pred = motion_submission_pb2.ChallengeScenarioPredictions()
            pred.scenario_id = data['id'][i]
            # print(pred.scenario_id)
            single_predictions = motion_submission_pb2.PredictionSet()

            for j in range(car_num):
                if not new_data['tracks_to_predict'][i, j]:
                    continue
                single_pred = motion_submission_pb2.SingleObjectPrediction()
                tmp = 0
                while new_data['misc'][i, j, tmp, 7] < 0.5:
                    tmp += 1
                single_pred.object_id = int(new_data['misc'][i, j, tmp, 8])

                yaw = new_data['misc'][i, j, 10, 4]

                coord[i, j] = rotate(coord[i, j], yaw)
                coord[i, j] += np.expand_dims(centroid[i, j], 0)
                coord[i, j] = rotate(coord[i, j], -1 * data['theta'][i])
                coord[i, j] += data['center'][i][np.newaxis, np.newaxis, :]
                for ki in range(K):
                    k = idx[i,j,ki]
                    scored_traj = motion_submission_pb2.ScoredTrajectory()

                    scored_traj.confidence = float(logit[i, j, k])
                    traj = motion_submission_pb2.Trajectory()
                    for ti in range(16):
                        current_time = 5 * ti + 4
                        traj.center_x.append(float(coord[i, j, k, current_time, 0]))
                        traj.center_y.append(float(coord[i, j, k, current_time, 1]))

                    scored_traj.trajectory.CopyFrom(traj)

                    single_pred.trajectories.append(scored_traj)

                single_predictions.predictions.append(single_pred)
            pred.single_predictions.CopyFrom(single_predictions)
            self.submission.scenario_predictions.append(pred)

        self.cnt += 1
        if self.cnt % 100 == 0:
            self.write()

    def write(self):
        if self.last_cnt == self.cnt:
            return

        dir = './submissions'
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir+f'/your_preds_{self.cnt}.bin', 'wb') as f:
            s = self.submission.SerializeToString()
            f.write(s)

        self.submission = motion_submission_pb2.MotionChallengeSubmission()

        self.submission.submission_type = 1
        self.submission.account_name = 'zqh10241024@gmail.com'
        self.submission.unique_method_name = 'mmTrans'
        self.last_cnt = self.cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--cfg', type=str, default='0')
    parser.add_argument('--model-name', type=str, default='default_model')
    args = parser.parse_args()

    cfg = load_config_data(f"./config/{args.cfg}.yaml")
    device = 'cpu' if args.local else 'cuda'
    if device == 'cpu':
        gpu_num = 1
        print('device: CPU')
    else:
        gpu_num = torch.cuda.device_count()
        print("gpu number:{}".format(gpu_num))
        print("gpu available:", torch.cuda.is_available())

    # print(cfg)
    dataset_cfg = cfg['dataset_cfg']
    dataset_cfg['dataset_dir'] = '/home/SENSETIME/fenglan/trans'
    train_dataset = WaymoDataset(dataset_cfg, 'validation')

    print('len:', len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))

    # =================================== INIT MODEL ============================================================
    model = load_model_class(cfg['model_name'])
    model_cfg = cfg['model_cfg']
    model = model(model_cfg)
    train_cfg = cfg['train_cfg']
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)
    model = torch.nn.DataParallel(model, list(range(gpu_num))) if args.local else torch.nn.DataParallel(model, list(
        range(gpu_num))).cuda()
    resume_model_name = os.path.join(
        'saved_models', '{}.pt'.format(args.model_name))
    model = load_checkpoint(resume_model_name, model, optimizer, args.local)
    print('Successful Resume model {}'.format(resume_model_name))


    submit = Submit()
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
            outputs_coord, outputs_class, new_data = model(data)
            output = {}
            output['pred_coords'] = outputs_coord
            output['pred_logits'] = outputs_class
            submit.fill(output, data, new_data)
    submit.write()
