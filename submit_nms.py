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

from utils.utilities import (load_checkpoint,load_model_class)
#from utils.traj_nms import traj_nms
def traj_nms(traj, score, threshold=2.0):
    '''
        traj:[K, 30, 2] list
        score:[K]
    '''
    sorted_index = np.argsort(score)[::-1]  # from max to min
    prop_num = 4
    candidate = []
    candidate_score = []

    while len(sorted_index):
        index = sorted_index[0]
        curr_traj = traj[index]
        candidate.append(curr_traj)
        candidate_score.append(score[index])
        if len(sorted_index) <= 1 or len(candidate) >= prop_num:
            break
        distance = np.linalg.norm(curr_traj[-1, :] - traj[sorted_index[1:]][:, -1, :], 2, axis=-1)
        new_index = np.where(distance > threshold)[0]  # the first one
        sorted_index = sorted_index[new_index + 1]

    K = len(candidate_score)
    candidate = np.array(candidate)
    candidate_score = np.array(candidate_score)
    candidate = np.pad(candidate,([0,prop_num-K],[0,0],[0,0]))
    score_sum = np.sum(candidate_score)
    candidate_score = -candidate_score/score_sum
    #candidate_score = torch.softmax(torch.Tensor(candidate_score),-1).numpy()
    candidate_score = np.pad(candidate_score, ([0, prop_num - K]))

    return candidate, candidate_score, K

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
        self.submission.account_name = 'lf2681@gmail.com'
        self.submission.unique_method_name = 'mmTrans'
        self.submission.authors.append('Lan Feng')
        self.submission.authors.append('Qihang Zhang')
        self.submission.affiliation = 'Sensetime & CUHK MMlab'

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
        for k in new_data.keys():
            new_data[k] = wash(new_data[k])
        for k in output.keys():
            output[k] = wash(output[k])
        coord = output['pred_coords']  # example: 32, 8, 6, 80, 2
        coord = coord.cumsum(-2)
        logit = output['pred_logits']
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

                candidate, candidate_score, K = traj_nms(coord[i,j], logit[i,j], 4)

                for ki in range(K):

                    scored_traj = motion_submission_pb2.ScoredTrajectory()

                    scored_traj.confidence = float(candidate_score[ki])
                    traj = motion_submission_pb2.Trajectory()
                    for ti in range(16):
                        current_time = 5 * ti + 4
                        traj.center_x.append(float(candidate[ki, current_time, 0]))
                        traj.center_y.append(float(candidate[ki, current_time, 1]))

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
        self.submission.account_name = 'lf2681@gmail.com'
        self.submission.unique_method_name = 'mmTrans'
        self.submission.authors.append('Lan Feng')
        self.submission.authors.append('Qihang Zhang')
        self.submission.affiliation = 'Sensetime & CUHK MMlab'
        self.last_cnt = self.cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action="store_true")
    # parser.add_argument('--cfg', type=str, default='0')
    # parser.add_argument('--model-name', type=str, default='default_model')
    args = parser.parse_args()

    cfg1 = 'candidate1'
    cfg2 = 'candidate2'
    cfg3 = 'candidate3'

    cfg1 = load_config_data(f"./config/{cfg1}.yaml")
    cfg2 = load_config_data(f"./config/{cfg2}.yaml")
    cfg3= load_config_data(f"./config/{cfg3}.yaml")

    device = 'cpu' if args.local else 'cuda'
    if device == 'cpu':
        gpu_num = 1
        print('device: CPU')
    else:
        gpu_num = torch.cuda.device_count()
        print("gpu number:{}".format(gpu_num))
        print("gpu available:", torch.cuda.is_available())

    # print(cfg)
    dataset_cfg = cfg1['dataset_cfg']
    dataset_cfg['dataset_dir'] = '/home/SENSETIME/fenglan/trans'
    train_dataset = WaymoDataset(dataset_cfg, 'validation')

    print('len:', len(train_dataset))

    train_dataloader = DataLoader(train_dataset, shuffle=dataset_cfg['shuffle'], batch_size=dataset_cfg['batch_size'],
                                  num_workers=dataset_cfg['num_workers'] * (not args.local))

    # =================================== INIT MODEL ============================================================
    model1 = load_model_class(cfg1['model_name'])
    model_cfg1 = cfg1['model_cfg']
    model1 = model1(model_cfg1)

    model2 = load_model_class(cfg2['model_name'])
    model_cfg2 = cfg2['model_cfg']
    model2 = model2(model_cfg2)

    model3 = load_model_class(cfg3['model_name'])
    model_cfg3 = cfg3['model_cfg']
    model3 = model3(model_cfg3)

    train_cfg = cfg1['train_cfg']

    optimizer = optim.AdamW(model1.parameters(), lr=train_cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_cfg['weight_decay'], amsgrad=True)

    model1 = torch.nn.DataParallel(model1, list(range(gpu_num))) if args.local else torch.nn.DataParallel(model1, list(
        range(gpu_num))).cuda()
    model2 = torch.nn.DataParallel(model2, list(range(gpu_num))) if args.local else torch.nn.DataParallel(model2, list(
        range(gpu_num))).cuda()
    model3 = torch.nn.DataParallel(model3, list(range(gpu_num))) if args.local else torch.nn.DataParallel(model3, list(
        range(gpu_num))).cuda()

    resume1 = os.path.join('saved_models', 'candidate1.pt')
    model1 = load_checkpoint(resume1, model1, optimizer, args.local)
    resume2 = os.path.join('saved_models', 'candidate2.pt')
    model2 = load_checkpoint(resume2, model2, optimizer, args.local)
    resume3 = os.path.join('saved_models', 'candidate3.pt')
    model3 = load_checkpoint(resume3, model3, optimizer, args.local)


    submit = Submit()
    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()
        progress_bar = tqdm(train_dataloader)
        cnt = 0
        for j, data in enumerate(progress_bar):
            for key in data.keys():
                if isinstance(data[key], torch.DoubleTensor):
                    data[key] = data[key].float()
                if isinstance(data[key], torch.Tensor) and not args.local:
                    data[key] = data[key].to('cuda:0')
            data1 = data.copy()
            outputs_coord1, outputs_class1, new_data,_ = model1(data1)
            data1 = data.copy()
            outputs_coord2, outputs_class2, new_data,_ = model2(data1)
            data1 = data.copy()
            outputs_coord3, outputs_class3, new_data,_ = model3(data1)

            coord = torch.cat([outputs_coord1,outputs_coord2,outputs_coord3],2)
            score = torch.cat([outputs_class1, outputs_class2, outputs_class3], 2)

            # coord = coord.reshape(-1,*coord.shape[2:])
            # score = score.reshape(-1, *score.shape[2:])
            # candidate = np.zeros([coord.shape[0],6,80,2])
            # candidate_score = np.zeros([coord.shape[0],6])
            # K = np.zeros(candidate.shape[0],dtype=int)
            # for i in range(coord.shape[0]):
            #     candidate[i],candidate_score[i],K[i] = traj_nms(coord[i],score[i],1.5)
            # candidate = candidate.reshape(*outputs_coord1.shape)
            # candidate_score = candidate_score.reshape(*outputs_class1.shape)
            # K = K.reshape(*outputs_coord1.shape[:2])

            output = {}
            output['pred_coords'] = coord
            output['pred_logits'] = score
            submit.fill(output, data, new_data)
    submit.write()
