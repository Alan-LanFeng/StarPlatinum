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
from lib.dataset.waymo_dataset import WaymoDataset
from l5kit.configs import load_config_data

from lib.models.STF.vectornet import VecNet
from lib.utils.utilities import (load_checkpoint, save_checkpoint, load_model_class,
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
        self.submission.submission_type = self.submission.SubmissionType.INTERACTION_PREDICTION
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
            # single_predictions = motion_submission_pb2.PredictionSet()
            joint_predictions = motion_submission_pb2.JointPrediction()

            yaw = new_data['misc'][i, :, 10, 4][:, np.newaxis, np.newaxis]

            coord[i, :] = rotate(coord[i, :], yaw)
            coord[i, :] += centroid[i, :][:, np.newaxis, np.newaxis, :]
            coord[i, :] = rotate(coord[i, :], -1 * data['theta'][i])
            coord[i, :] += data['center'][i][np.newaxis, np.newaxis, :]
            for k in range(K):
                scored_joint_traj = motion_submission_pb2.ScoredJointTrajectory()
                scored_joint_traj.confidence = float(logit[i,k]) + 100
                for j in range(car_num):
                    if not new_data['tracks_to_predict'][i,j]:
                        continue
                    obj_traj = motion_submission_pb2.ObjectTrajectory()
                    tmp = 0
                    while new_data['misc'][i, j, tmp, 7] < 0.5:
                        tmp += 1
                    obj_traj.object_id = int(new_data['misc'][i,j,tmp, 8])
                    traj = motion_submission_pb2.Trajectory()
                    for ti in range(16):
                        current_time = 5 * ti + 4
                        traj.center_x.append(float(coord[i, j, k, current_time, 0]))
                        traj.center_y.append(float(coord[i, j, k, current_time, 1]))
                    obj_traj.trajectory.CopyFrom(traj)
                    scored_joint_traj.trajectories.append(obj_traj)
                joint_predictions.joint_trajectories.append(scored_joint_traj)
            pred.joint_prediction.CopyFrom(joint_predictions)
            self.submission.scenario_predictions.append(pred)

        self.cnt += 1
        if self.cnt % 100 == 0:
            self.write()

    def write(self):
        if self.last_cnt == self.cnt:
            return

        with open(f'/tmp/models/your_preds_{self.cnt}.bin', 'wb') as f:
            s = self.submission.SerializeToString()
            f.write(s)

        self.submission = motion_submission_pb2.MotionChallengeSubmission()

        self.submission.submission_type = 2
        self.submission.account_name = 'zqh10241024@gmail.com'
        self.submission.unique_method_name = 'mmTrans'
        self.last_cnt = self.cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--training-tricks', action="store_true")
    parser.add_argument('--train-in-validation', action="store_true")
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--data-augment', action="store_true")
    parser.add_argument('--cfg-name', type=str, default='argoverse')
    parser.add_argument('--debug-mode', action="store_true")
    parser.add_argument('--model-name', type=str, default='defualt_model')
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--waymo-dir', type=str, default='/mnt/lustre/share/zhangqihang/WOD/trans')

    args = parser.parse_args()
    argoverse = True
    if argoverse:
        TRAIN_DIR = os.path.join('./intermediate_data', 'train_intermediate')
        TRIAN_DATA_PKL = 'Train_data_flip_Feature.pkl' if args.data_augment else 'Features.pkl'
        if args.debug_mode:
            TRIAN_DATA_PKL = 'Features.pkl'
            TRAIN_DIR = os.path.join('./intermediate_data', 'sample_intermediate')
    # =================Get Config================================================================================
    config_file_name = 'agent_motion_config' if not argoverse else args.cfg_name
    cfg = load_config_data(f"./config/{config_file_name}.yaml")
    cfg['local'] = args.local
    device = 'cuda'
    # print(cfg)
    model_params = cfg['model_params']
    train_params_cfg = cfg['train_params']
    gpu_num = torch.cuda.device_count()
    print("gpu number:{}".format(gpu_num))
    print(torch.cuda.is_available())
    # ================================== INIT DATASET ==========================================================
    start_time = time.time()
    DATA_CFG_NAME_TRAIN = 'train_data_loader'
    # DATA_CFG_NAME_TRAIN = 'sample_data_loader'
    train_cfg = cfg[DATA_CFG_NAME_TRAIN]
    cfg['train_params']['is_augment'] = args.data_augment
    train_cfg['local'] = args.local

    train_dataset = WaymoDataset(root=args.waymo_dir, period='validation_interactive')
    print('len:', len(train_dataset))
    collate_fn = None

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8,
                                  num_workers=train_cfg["num_workers"], collate_fn=collate_fn)

    # ============================= Some Parameter Initial =====================================================
    max_epoch = train_params_cfg['num_epoch']
    future_frames_num = cfg['model_params']['future_num_frames']
    save_freq = train_params_cfg['save_freq']
    data_time, model_time = 0, 0
    dataset_len = len(train_dataloader)
    best_MR = 1.0
    # =================================== INIT MODEL ============================================================
    model_save_path = './models/'
    model_cfg = cfg['model_params']
    model_cfg['local'] = args.local
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    module_name = cfg['model_name']
    STF = load_model_class(module_name)
    STF = STF(cfg['model_params'])
    model = VecNet(STF, model_cfg, train_cfg['lane_length'], device)

    if 'backbone_slow' in train_params_cfg and train_params_cfg['backbone_slow']:
        filter_list = ['cls_partition_mlp', 'cls_mlp']
        back_bone_params = nn.ParameterList()
        unfreeze_params = nn.ParameterList()
        unfreeze_params_name = []
        back_bone_params_name = []


        def filter_params(params, filter_list):

            for name, p in params:

                flag = 0
                for f in filter_list:
                    if f in name:
                        unfreeze_params.append(p)
                        flag = 1
                        break
                if not flag:
                    back_bone_params.append(p)


        filter_params(model.named_parameters(), filter_list)
        params = [{'params': back_bone_params, 'lr': 0.00001},
                  {'params': unfreeze_params,
                   'lr': train_params_cfg['learning_rate']},
                  ]
    else:
        params = [{'params': model.parameters()}, ]

    log_vars = None
    if cfg['loss_params']['MultiTask']:
        log_vars = nn.ParameterList()
        for i in range(4):
            log_var = torch.tensor([1.0])
            log_var = nn.Parameter(log_var, requires_grad=True)
            log_vars.append(log_var)
        log_vars = log_vars.to(device)
        params.append({'params': log_vars})

    optimizer = optim.AdamW(params, lr=train_params_cfg['learning_rate'],
                            betas=(0.9, 0.999), eps=1e-09,
                            weight_decay=train_params_cfg['weight_decay'],
                            amsgrad=True)
    if args.training_tricks:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, train_params_cfg['restart_epoch'], T_mult=2, last_epoch=-1)
    else:
        step_size = train_params_cfg['decay_lr_every_epoch'] * \
                    dataset_len if argoverse else train_params_cfg['decay_lr_every_iter']
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=train_params_cfg['decay_lr_factor'])

    model = torch.nn.DataParallel(model, list(range(gpu_num))) if args.local else torch.nn.DataParallel(model, list(
        range(gpu_num))).cuda()
    if args.resume:
        resume_model_name = os.path.join(
            model_save_path, '{}.pt'.format(args.model_name))
        # resume_model_name = os.path.join(model_save_path,'{}_{}.pt'.format(args.model_name, train_params_cfg[
        # 'resume_epoch']))
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

            # if 'ccb809abe7e3b4b6' not in data['id']:
            #     continue
            output, new_data = model(data)
            submit.fill(output, data, new_data)
    submit.write()
