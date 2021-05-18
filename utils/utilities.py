import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import math


def save_checkpoint(checkpoint_dir, model, optimizer, MR=1.0):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizerâ€™s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'BestMissRate': MR
    }

    torch.save(state, checkpoint_dir)
    print('model saved to %s' % checkpoint_dir)


def load_checkpoint(checkpoint_path, model, optimizer=None, local=False):
    if not local:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # new_dict = {k[7:]:v for k,v in state['state_dict'].items()}
    model.load_state_dict(state['state_dict'])
    # if not optimizer is None:
    #     optimizer.load_state_dict(state['optimizer'])

    print('model loaded from %s' % checkpoint_path)
    return model


def vis_before_feed(data: dict,
                    obs_len=26,
                    fut_len=50):
    lane = data['lane']
    hist = data['history_positions']
    agent_pos = data['target_centroid']
    gt = data['groundtruth']
    gt_available = data['groundtruth_availabilities']
    nbrs = data['nbrs_history_feature']
    nbrs_gt = data['nbrs_groundtruth']
    nbrs_centroid = data['neighbour_centroid']

    lane = lane
    hist = hist.numpy()
    centroid = agent_pos.numpy()
    gt = gt.numpy().reshape(fut_len, 2)
    nbrs = nbrs
    nbrs_gt = nbrs_gt
    nbrs_centroid = nbrs_centroid

    fig = plt.figure()
    ax = plt.subplot()
    # draw lane
    ax = draw_lane(ax, lane, centroid)
    # draw agent
    # history 
    hist_traj = (hist[..., :2].cumsum(axis=0) + centroid)[slice(max(np.nonzero(hist[..., 2])[0]) + 1)]
    hist_traj = np.concatenate((centroid[np.newaxis, :], hist_traj), axis=0)
    # ax.plot(hist_traj[:,0],hist_traj[:,1], '-', color='green', linewidth=3, zorder=2)
    ax = draw_ego_agents(ax, hist_traj, color='green')
    future_traj = (gt + centroid)[:np.nonzero(gt_available)[-1] + 1]
    future_traj = np.concatenate((centroid[np.newaxis, :], future_traj), axis=0)
    ax = draw_ego_agents(ax, future_traj, color='blue')
    # draw nbrs
    for j in range(nbrs.shape[0]):
        polyline = nbrs[j]
        # print(polyline[:,[0,1,3]])
        hist_available_interval = len(np.nonzero(polyline[..., 3]))
        if hist_available_interval:
            hist_traj_nbrs = (polyline[..., :2].cumsum(axis=0) + centroid)[:hist_available_interval]
            ax = draw_ego_agents(ax, hist_traj_nbrs, color='red')
        gt_nbrs = nbrs_gt[j, :, :2]
        future_available_interval = len(np.nonzero(nbrs_gt[j, :, 2])[0])
        if future_available_interval:
            future_traj_nbrs = (gt_nbrs.cumsum(axis=0) + centroid)[:future_available_interval]
            ax = draw_ego_agents(ax, future_traj_nbrs, color='pink')

    ax.axis('off')
    fig.show()


def draw_ego_agents(ax, trajectory, color: str = 'blue', zorder=2):
    ax.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, linewidth=3, zorder=zorder)
    return ax


def draw_lane(ax, lane, centroid):
    for i in range(len(lane)):
        polyline = lane[i, :, :]
        coords = np.vstack([polyline[:, :2], polyline[-1, 2:4]]) + centroid
        ax.plot(coords[:, 0], coords[:, 1], '--', color='grey', linewidth=1)
    return ax


def vis_argoverse(data, idx, obs_len=20, fut_len=30, pred=None, name=None):
    lane = data['lane'][idx]
    hist = data['history_positions'][idx]
    agent_pos = data['target_centroid'][idx]
    gt = data['groundtruth'][idx]
    # gt_available=data['groundtruth_availabilities']
    nbrs = data['nbrs_history_feature'][idx]
    nbrs_gt = data['nbrs_groundtruth'][idx]
    nbrs_pos = data['neighbour_centroid'][idx]

    lane = lane.detach().cpu().numpy()
    hist = hist.detach().cpu().numpy()
    agent_pos = agent_pos.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy().reshape(fut_len, 2)
    nbrs = nbrs.detach().cpu().numpy()
    nbrs_gt = nbrs_gt.detach().cpu().numpy()
    nbrs_pos = nbrs_pos.detach().cpu().numpy()
    # good_nbrs = good_nbrs.detach().cpu().numpy()
    # draw lane
    for i in range(lane.shape[0]):
        polyline = lane[i, :, :]
        coords = np.vstack([polyline[:, :2], polyline[-1, 2:4]])
        if (polyline[0][5] == 2):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='black', linewidth=1)
        elif (polyline[0][5] == 3):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='purple', linewidth=1)
        else:
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='grey', linewidth=1)

    # draw agent
    coords = np.zeros((obs_len, 2))
    coords[obs_len - 1] = agent_pos
    for i in range(len(hist)):
        coords[obs_len - (i + 1) - 1] = coords[obs_len - (i) - 1] - hist[len(hist) - 1 - i, :2]
    b, = plt.plot(coords[:, 0], coords[:, 1], '-', color='green', linewidth=3, zorder=2)

    coords = np.zeros((len(gt) + 1, 2))
    coords[0] = agent_pos
    for i in range(len(gt)):
        temp = coords[i] + gt[i]
        coords[i + 1] = temp
    c, = plt.plot(coords[:, 0], coords[:, 1], '-', color='blue', linewidth=3, zorder=2)

    # draw nbrs
    for j in range(nbrs.shape[0]):
        polyline = nbrs[j, :, :]
        # print(polyline[:,[0,1,3]])
        coords = np.zeros((obs_len, 2))
        coords[obs_len - 1] = nbrs_pos[j]
        for i in range(len(polyline)):
            coords[obs_len - (i + 1) - 1] = coords[obs_len - (i) - 1] - polyline[len(polyline) - 1 - i, :2]
        d, = plt.plot(coords[:, 0], coords[:, 1], '-', color='red', linewidth=1, zorder=2)

        gt = nbrs_gt[j, :, :2]
        coords = np.zeros((len(gt) + 1, 2))
        coords[0] = nbrs_pos[j]
        for i in range(len(gt)):
            temp = coords[i] + gt[i]
            coords[i + 1] = temp
        e, = plt.plot(coords[1:, 0], coords[1:, 1], '-', color='pink', linewidth=1, zorder=2)
    # print(good_nbrs)

    # plt.axis('off')
    # plt.show()
    # import pdb; pdb.set_trace()
    # plt.savefig(f'./{name}.jpg')
    # plt.close()


def vis_argoverse_nogt(data, idx, obs_len=20, fut_len=30, pred=None):
    lane = data['lane'][idx]
    hist = data['history_positions'][idx]
    agent_pos = data['target_centroid'][idx]
    # gt = data['groundtruth'][idx]
    # gt_available=data['groundtruth_availabilities']
    nbrs = data['nbrs_history_feature'][idx]
    # nbrs_gt = data['nbrs_groundtruth'][idx]
    nbrs_pos = data['neighbour_centroid'][idx]

    lane = lane.detach().cpu().numpy()
    hist = hist.detach().cpu().numpy()
    agent_pos = agent_pos.detach().cpu().numpy()
    # gt = gt.detach().cpu().numpy().reshape(fut_len,2)
    nbrs = nbrs.detach().cpu().numpy()
    # nbrs_gt = nbrs_gt.detach().cpu().numpy()
    nbrs_pos = nbrs_pos.detach().cpu().numpy()
    # good_nbrs = good_nbrs.detach().cpu().numpy()
    # draw lane
    for i in range(lane.shape[0]):
        polyline = lane[i, :, :]
        coords = np.vstack([polyline[:, :2], polyline[-1, 2:4]])
        if (polyline[0][5] == 2):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='black', linewidth=1)
        elif (polyline[0][5] == 3):
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='purple', linewidth=1)
        else:
            a, = plt.plot(coords[:, 0], coords[:, 1], '--', color='grey', linewidth=1)

    # draw agent
    coords = np.zeros((obs_len, 2))
    coords[obs_len - 1] = agent_pos
    for i in range(len(hist)):
        coords[obs_len - (i + 1) - 1] = coords[obs_len - (i) - 1] - hist[len(hist) - 1 - i, :2]
    b, = plt.plot(coords[:, 0], coords[:, 1], '-', color='green', linewidth=3, zorder=2)

    # coords = np.zeros((len(gt)+1,2))
    # coords[0] = agent_pos
    # for i in range(len(gt)):
    #     temp=coords[i] + gt[i]
    #     coords[i+1]=temp
    # c, =plt.plot(coords[:,0],coords[:,1], '-', color='blue', linewidth=3, zorder=2)

    # draw agent
    pred = pred[idx][0]
    for ii in range(len(pred)):
        coords = pred[ii]
        coords = coords + agent_pos

        c, = plt.plot(coords[:, 0], coords[:, 1], '-', color='navy', linewidth=3, zorder=1)

    # draw nbrs
    for j in range(nbrs.shape[0]):
        polyline = nbrs[j, :, :]
        # print(polyline[:,[0,1,3]])
        coords = np.zeros((obs_len, 2))
        coords[obs_len - 1] = nbrs_pos[j]
        for i in range(len(polyline)):
            coords[obs_len - (i + 1) - 1] = coords[obs_len - (i) - 1] - polyline[len(polyline) - 1 - i, :2]
        d, = plt.plot(coords[:, 0], coords[:, 1], '-', color='red', linewidth=1, zorder=2)

        # gt = nbrs_gt[j,:,:2]
        # coords = np.zeros((len(gt)+1,2))
        # coords[0] = nbrs_pos[j]
        # for i in range(len(gt)):
        #     temp=coords[i] + gt[i]
        #     coords[i+1]=temp
        # e, =plt.plot(coords[1:,0],coords[1:,1], '-', color='pink', linewidth=1, zorder=2)
    # print(good_nbrs)

    plt.axis('off')
    # plt.savefig('./111.jpg')
    plt.show()


def shuffle_index(old_data):
    data = old_data.copy()
    hist = data['history_positions']
    ag_pos = data['target_centroid']
    valid_len = data['vaild_len']
    nbrs = data['nbrs_history_feature']
    nbrs_pos = data['neighbour_centroid']
    gt = data['groundtruth']
    nbrs_gt = data['nbrs_groundtruth']
    max_len = data['max_len']
    gt_mask = data['groundtruth_availabilities']

    valid_len = valid_len.view(-1, 3)
    max_lane_num = max_len[0, 0].int()
    max_nbrs_num = max_len[0, 1].int()
    batch_size = hist.shape[0]
    # traj
    hist = hist.view(batch_size, 1, 19, 4)
    nbrs = nbrs.view(batch_size, max_nbrs_num, 19, 4)
    traj = torch.cat([hist, nbrs], dim=1)

    # pos
    ego_pos = ag_pos.view(-1, 1, 2)
    nbrs_pos = nbrs_pos.view(-1, max_nbrs_num, 2)
    pos = torch.cat([ego_pos, nbrs_pos], 1)

    # all_gt
    gt = gt.view(batch_size, 1, 30, 2)
    gt_mask = gt_mask.view(batch_size, 1, 30, 1)
    gt = torch.cat((gt, gt_mask), dim=-1)
    nbrs_gt = nbrs_gt.view(batch_size, max_nbrs_num, 30, 3)
    all_gt = torch.cat((gt, nbrs_gt), dim=1)
    shuffle_idxs = []
    for batch in range(batch_size):
        batch_valid_len = valid_len[batch, 1] + 1
        index = np.arange(batch_valid_len.detach().cpu().numpy())
        shuffle_idx = index.copy()
        np.random.shuffle(shuffle_idx)
        pos[batch][:batch_valid_len] = pos[batch][shuffle_idx]
        traj[batch][:batch_valid_len] = traj[batch][shuffle_idx]
        all_gt[batch][:batch_valid_len] = all_gt[batch][shuffle_idx]
        shuffle_idxs.append(shuffle_idx)

    data['history_positions'] = traj[:, 0]
    data['target_centroid'] = pos[:, 0]
    data['nbrs_history_feature'] = traj[:, 1:]
    data['neighbour_centroid'] = pos[:, 1:]
    data['groundtruth'] = all_gt[:, :1, ..., :2]
    data['groundtruth_availabilities'] = all_gt[:, :1, ..., 2]
    data['nbrs_groundtruth'] = all_gt[:, 1:]
    return data, shuffle_idxs


def check_output(data, model, criterion):
    with torch.no_grad():
        new_data, shuffle_idxs = shuffle_index(data)
        target_pred = model(data)
        target_pred_new = model(new_data)
        _, _, _, MR = criterion(target_pred, data)
        for i in range(len(shuffle_idxs)):
            target_pred[0][i][:len(shuffle_idxs[i])] = target_pred[0][i][shuffle_idxs[i]]

        _, _, _, MR_cp = criterion(target_pred, new_data)
        _, _, _, MR_new = criterion(target_pred_new, new_data)
        import pdb;
        pdb.set_trace()
        if MR == MR_new:
            return True
        else:
            return False


def load_model_class(model_name):
    import importlib
    module_path = f'models.{model_name}'
    target_module = importlib.import_module(module_path)
    target_class = getattr(target_module, model_name)
    return target_class


def get_displacement_errors_and_miss_rate_vector(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        max_guesses: int,
        horizon: int,
        miss_threshold: float,
        forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade = [], []
    min_fde, prob_min_fde = [], []
    n_misses, prob_n_misses = [], []

    traj = np.array([i for k, i in forecasted_trajectories.items()])
    gt = np.array([i for k, i in gt_trajectories.items()])

    traj_fd = traj[:, :, -1, :]
    gt_fd = np.expand_dims(gt, axis=1)[:, :, -1, :2]
    mask = gt[:, -1, -1].astype(np.bool)
    diff = np.sum((gt_fd - traj_fd) ** 2, axis=-1) ** 0.5
    mindiff = np.min(diff[mask], axis=-1)

    metric_results["minFDE"] = np.mean(mindiff)
    metric_results["MR"] = np.mean(mindiff > miss_threshold)

    return metric_results


def get_displacement_errors_and_miss_rate(
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        max_guesses: int,
        horizon: int,
        miss_threshold: float,
        forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade = [], []
    min_fde, prob_min_fde = [], []
    n_misses, prob_n_misses = [], []
    cnt = 0
    for k, v in gt_trajectories.items():

        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)

    return metric_results


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def fix_parameter_except(model: torch.nn.Module, sub_model_name: dict = None):
    for name, child_model in model.named_children():
        if name in sub_model_name:
            set_model_grad(child_model, True)


def set_model_grad(model: torch.nn.Module, require_grad=False):
    for p in model.parameters():
        p.requires_grad = require_grad
