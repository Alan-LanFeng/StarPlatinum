import os, sys
import time

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .waymo_dataset import WaymoDataset

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class WODEvaluator(object):
    def __init__(self, cfg, device):
        dataset_cfg = cfg['dataset_cfg']
        val_dataset = WaymoDataset(dataset_cfg, 'validation')
        self.val_dataloader = DataLoader(val_dataset,
                                         shuffle=dataset_cfg['shuffle'],
                                         batch_size=dataset_cfg['batch_size'] if device != 'cpu' else 4,
                                         num_workers=dataset_cfg['num_workers'],
                                         collate_fn=None)
        self.best_miss_rate = 1.0
        self.device = device
        self.metric_default_cfg = self._get_default_config()
        self.mode = cfg['track']

    @staticmethod
    def _get_default_config():
        cfg = motion_metrics_pb2.MotionMetricsConfig()
        config_text = """
          track_steps_per_second: 10
          prediction_steps_per_second: 2
          track_history_samples: 10
          track_future_samples: 80
          speed_lower_bound: 1.4
          speed_upper_bound: 11.0
          speed_scale_lower: 0.5
          speed_scale_upper: 1.0
          step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
          }
          step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
          }
          step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
          }
          max_predictions: 6
          """
        text_format.Parse(config_text, cfg)
        return cfg.SerializeToString()

    def get_res_in_tabular(self, res, verbose=True):

        rows = ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'mAP']
        columns = ['vehi_5', 'vehi_9', 'vehi_15', 'ped_5', 'ped_9', 'ped_15', 'cyc_5', 'cyc_9', 'cyc_15']
        ret = {}

        if verbose:
            mag = 96
            print(' ' + '-' * (mag))
            print('|' + ' ' * 15 + '|' + ''.join(["%8s|" % columns[i] for i in range(9)]))
            print(' ' + '-' * (mag))
        for i in range(5):
            out = f"|%15s|" % (rows[i])
            for j in range(9):
                v = res[i][j]
                out += f"%8.4f|" % v
                ret[rows[i] + '/' + columns[j]] = v
            if verbose:
                print(out)
                print('-' * (mag))
        return ret

    def evaluate(self, model):
        assert self.mode in ['motion', 'interaction'], 'mode should be motion or interaction'

        model.eval()
        progress_bar = tqdm(self.val_dataloader)

        t = tf.convert_to_tensor
        T = lambda x: t(x.cpu().detach().numpy())

        tot = [[0 for _ in range(9)] for __ in range(5)]

        with torch.no_grad():
            cnt = [0, 0, 0]
            for i, data in enumerate(progress_bar):
                for key in data.keys():
                    try:
                        data[key] = data[key].to(self.device)
                    except:
                        pass
                coord, score, new_data = model(data)

                coord = coord.detach()
                score = score.detach()

                predict_flag = new_data['tracks_to_predict']
                yaw = new_data['misc'][..., 10, 4]
                centroid = new_data['centroid']

                interval = 5

                if self.mode == 'motion':
                    def select(x):
                        return x[predict_flag].unsqueeze(1)

                    pred = select(coord).permute(0, 2, 1, 3, 4).cumsum(
                        -2)  # batch_size, K, car_num, horizon, 2
                    batch_size, K, car_num, horizon, c = pred.shape
                    if batch_size * car_num == 0:
                        continue

                    yaw = select(yaw)
                    centroid = select(centroid)
                    s, c = torch.sin(yaw), torch.cos(yaw)
                    s, c = s.view(batch_size, 1, 1, 1), c.view(batch_size, 1, 1, 1)
                    pred[..., 0], pred[..., 1] = c * pred[..., 0] - s * pred[..., 1], \
                                                 s * pred[..., 0] + c * pred[..., 1]
                    pred += centroid.view(batch_size, 1, 1, 1, 2)
                    pred = pred[..., (interval - 1)::interval, :]

                    misc = select(new_data['misc'])
                    gt = misc[..., :7]
                    gt_is_valid = misc[..., 7] > 0
                    object_type = select(new_data['obj_type'])
                    # pred = pred.permute(0, 2, 1, 3, 4).reshape(batch_size * car_num, 1, K, -1, c).permute(0, 2, 1, 3, 4)
                    gt = gt.reshape(batch_size * car_num, 1, 91, 7)
                    gt_is_valid = gt_is_valid.reshape(batch_size * car_num, 1, 91)
                    object_type = object_type.reshape(batch_size * car_num, 1)
                    score = select(score)
                    score = score.reshape(batch_size * car_num, -1)
                else:
                    predict_flag = predict_flag.sum(-1) == 2
                    if predict_flag.sum() == 0:
                        continue

                    def select(x):
                        return x[predict_flag]

                    misc = select(new_data['misc'])
                    gt = misc[..., :7]
                    gt_is_valid = misc[..., 7] > 0
                    object_type = select(new_data['obj_type'])
                    score = select(score)

                    idx = (-1*object_type).argsort(1)
                    pred = torch.gather(pred, 2, idx.view(idx.shape[0], 1, idx.shape[1], 1, 1).repeat(1, pred.shape[1], 1, *pred.shape[-2:]))
                    gt = torch.gather(gt, 1, idx.view(*idx.shape, 1, 1).repeat(1,1,*gt.shape[-2:]))
                    gt_is_valid = torch.gather(gt_is_valid, 1, idx.view(*idx.shape, 1).repeat(1,1,gt_is_valid.shape[-1]))
                    object_type = torch.gather(object_type, 1, idx)

                with suppress_stdout_stderr():
                    x1, x2, x3, x4, x5 = T(pred), T(score), T(gt), T(gt_is_valid), T(object_type)
                    res = py_metrics_ops.motion_metrics(config=self.metric_default_cfg,
                                                        prediction_trajectory=x1,
                                                        prediction_score=x2,
                                                        ground_truth_trajectory=x3,
                                                        ground_truth_is_valid=x4,
                                                        object_type=x5)
                resl = [res.min_ade, res.min_fde, res.miss_rate, res.overlap_rate, res.mean_average_precision]
                assert 1 == 1
                for j in range(5):
                    for k in range(3):
                        if self.mode == 'motion':
                            showtime = ((object_type == k + 1) * (gt_is_valid.sum(-1) > 0)).sum()
                        else:
                            showtime = ((object_type[:, 0] == k + 1) * (gt_is_valid.sum(-1) > 0)).sum()

                        if showtime == 0:
                            continue
                        for time in range(3):
                            resl_numpy = resl[j][3 * k + time].numpy()
                            assert resl_numpy >= 0, f'{showtime}-{j}-{k}-{time}-{resl_numpy}-{object_type}'
                            tot[j][3 * k + time] = (tot[j][3 * k + time] * cnt[k] + resl_numpy) / (cnt[k] + 1)
                        cnt[k] += 1

        return self.get_res_in_tabular(tot, verbose=True)