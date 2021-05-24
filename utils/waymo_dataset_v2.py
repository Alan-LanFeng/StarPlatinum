from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from l5kit.configs import load_config_data

MAX_LANE_NUM = 660
MAX_NBRS_NUM = 127
MAX_AGENT_NUM = 128
FUTURE_LEN = 80
TIMEFRAME = 91
CURRENT = 10
LANE_SAMPLE = 10

# adjacent lane params
# any lane in this range will become a specific agents' adjacent lane
DIST = 15  # 10m in front of the agent
RADIUS = 20  # circle with radius 15

import os


class WaymoDataset(Dataset):

    def __init__(self, cfg, period):
        super(WaymoDataset, self).__init__()

        periods = ['testing', 'testing_interactive', 'training', 'validation', 'validation_interactive']
        assert period in periods

        self.root = cfg['dataset_dir']
        self.period = period
        self.path = os.path.join(self.root, period)
        self.cache = cfg['cache']
        self.cache_name = cfg['cache_name']
        self.shrink = cfg['shrink']

    def __len__(self):
        with open(os.path.join(self.root, 'len.txt'), 'r') as f:
            l = f.read().split()
        i = 0
        while l[i] != self.period:
            i = i + 2
        return int(l[i + 1]) // 5 if self.shrink else int(l[i + 1])

    def __getitem__(self, index):
        if self.cache:
            cache_root = self.root[:self.root.find('trans')]
            cache_file = os.path.join(cache_root, self.cache_name, self.period, f'{index}.pkl')
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            file_path = os.path.join(self.path, f'{index}.pkl')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return self.process(data)

    def hist_process(self, traj):
        vector = np.concatenate([traj[..., :CURRENT, :2], traj[..., 1:CURRENT + 1, :2]], -1)
        mask = traj[..., :CURRENT, -2] * traj[..., 1:CURRENT + 1, -2]
        return np.pad(vector, [(0, MAX_AGENT_NUM - vector.shape[0]), (0, 0), (0, 0)]), \
               np.pad(mask, [(0, MAX_AGENT_NUM - mask.shape[0]), (0, 0)]).astype(bool)

    # TODO: try control signal, try rho theta
    def gt_process(self, traj):
        future = traj[..., CURRENT:, :]
        centroid = np.expand_dims(traj[..., CURRENT, :2], 1).repeat(future.shape[-2], 1)
        ego_centric = future[..., :2] - centroid
        yaw = future[..., 0, 5]
        ego_centric = self.rotate(ego_centric, yaw)
        vector = ego_centric[..., 1:, :2] - ego_centric[..., :-1, :2]
        mask = future[..., 1:, -2] * future[..., :-1, -2]
        cum_mask = mask.cumsum(-1) == np.ones_like(mask).cumsum(-1)
        return np.pad(vector, [(0, MAX_AGENT_NUM - vector.shape[0]), (0, 0), (0, 0)]), \
               np.pad(cum_mask, [(0, MAX_AGENT_NUM - cum_mask.shape[0]), (0, 0)])

    def lane_process(self, lane,traf):

        # first divide [20000,9] point data into [660,10,4] lane data, each lane have 10 points
        lane_vector = np.zeros([660,9,11],dtype=np.float32)
        rid = lane[:, 0]
        id_set = np.unique(rid)
        cnt = 0

        traf_id = traf[:, 0]
        for _id in id_set:
            if _id == -1:
                continue

            ind = np.argwhere(traf_id == _id)
            if ind.shape[0] == 0:
                ts = [0,0,0]
            else:
                ts = traf[ind[0][0],-2]
                if ts in [1,4,7]:
                    ts = [1,0,0]
                elif ts in[2,5,8]:
                    ts = [0,1,0]
                elif ts in [3,6]:
                    ts = [0,0,1]
                else:
                    ts = [0,0,0]

            arg = np.where(rid == _id)[0]
            # feature includes x,y,type,validity, id
            lis = [1, 2]
            feat = lane[arg, :]
            # filter lane type white/yellow lane
            ty = feat[0, 7]
            if ty in range(6, 17):
                continue
            if ty in range(1, 4):
                #feat[:,7] = 1
                one_hot = [1,0,0,0]
            if ty in range(17,20):
                if ty == 17:
                    one_hot = [0,1,0,0]
                elif ty ==18:
                    one_hot = [0,0,1,0]
                elif ty == 19:
                    one_hot = [0,0,0,1]
            all_one_hot = one_hot + ts
            valid = feat[:, 8]
            feat = feat[:, lis]
            feat = feat[valid==1]
            point_num = len(arg)
            if point_num <= LANE_SAMPLE:
                if point_num==1:
                    lane_vector[cnt,0,:2] = feat[0,:2]
                    lane_vector[cnt, 0, 2:4] = feat[0,:2]
                    lane_vector[cnt,0,4:] = all_one_hot
                else:
                    lane_vector[cnt,:point_num-1,:2] = feat[:-1,:2]
                    lane_vector[cnt, :point_num-1, 2:4] = feat[1:, :2]
                    lane_vector[cnt, :point_num-1, 4:] = all_one_hot
                    if point_num<10 and ty in [18,19]:
                        lane_vector[cnt,point_num-1,:2] = lane_vector[cnt,point_num-2,2:4]
                        lane_vector[cnt, point_num - 1, 2:4] = lane_vector[cnt, 0, :2]
                        lane_vector[cnt, point_num - 1, 4:] = all_one_hot
            else:
                interval = 1.0 * (point_num - 1) / (LANE_SAMPLE - 1)
                selected_point_index = [int(np.round(i * interval)) for i in range(1, LANE_SAMPLE - 1)]
                selected_point_index = [0] + selected_point_index + [point_num - 1]
                selected_point = feat[selected_point_index, :]
                lane_vector[cnt,:,:2] = selected_point[:-1,:]
                lane_vector[cnt, :, 2:4] = selected_point[1:, :]
                lane_vector[cnt,:,4:] = all_one_hot
            cnt+=1
        valid_len = cnt
        return lane_vector, valid_len

    def map_allocation(self, xy, theta, lane):

        front_x, front_y = xy[:, 0] - np.sin(theta) * DIST, xy[:, 1] + np.cos(theta) * DIST
        lane_xy = (lane[..., :2] + lane[..., 2:4]) / 2
        mask = np.sum(lane_xy,-1)==0
        lane_xy[mask, :] = 100000
        lane_xy = lane_xy.reshape(-1, 2)
        lane_x, lane_y = lane_xy[:, 0], lane_xy[:, 1]

        front_x = front_x[:, np.newaxis].repeat(len(lane_x), axis=1)
        front_y = front_y[:, np.newaxis].repeat(len(lane_y), axis=1)
        lane_x = lane_x[np.newaxis, :].repeat(len(front_x), axis=0)
        lane_y = lane_y[np.newaxis, :].repeat(len(front_y), axis=0)

        dist = np.sqrt(np.square(front_x - lane_x) + np.square(front_y - lane_y)).reshape(len(front_x), -1,
                                                                                          lane.shape[1])
        dist_min = np.min(dist, -1)
        adj = dist_min < RADIUS
        adj_index, adj_mask = self.bool2index(adj)

        return np.pad(adj_index, [(0, MAX_AGENT_NUM - adj_index.shape[0]), (0, MAX_LANE_NUM - adj_index.shape[1])]) \
            , np.pad(adj_mask, [(0, MAX_AGENT_NUM - adj_mask.shape[0]), (0, MAX_LANE_NUM - adj_mask.shape[1])])

    def rotate(self, traj, yaw):
        c, s = np.cos(yaw)[:, np.newaxis], np.sin(yaw)[:, np.newaxis]
        traj[..., 0], traj[..., 1] = c * traj[..., 0] + s * traj[..., 1], -s * traj[..., 0] + c * traj[..., 1]
        return traj

    # transform bool mask to index for the purpose of gather
    def bool2index(self, mask):
        seq = range(mask.shape[-1])
        index = mask * seq
        index[mask == 0] = 1000
        index = np.sort(index, -1)
        mask = index < 1000
        index[index == 1000] = 0
        return index, mask

    def process(self, data):
        out = dict()
        # since there's no id in cluster,so we fabricate it
        data['ego_p_c_f'] = np.pad(data['ego_p_c_f'], [(0, 0), (0, 13 - data['ego_p_c_f'].shape[-1])])
        data['nbrs_p_c_f'] = np.pad(data['nbrs_p_c_f'], [(0, 0), (0, 0), (0, 13 - data['nbrs_p_c_f'].shape[-1])])

        # traj
        data['ego_p_c_f'] = np.expand_dims(data['ego_p_c_f'], 0)
        all_traj = np.concatenate([data['ego_p_c_f'], data['nbrs_p_c_f']], 0)

        misc_list = [0, 1, 6, 7, 5, 3, 4, 11, 12]
        out['misc'] = all_traj[:, :, misc_list]

        current_valid_index = all_traj[..., CURRENT, -2] == 1
        valid_agent_num = sum(current_valid_index)
        all_traj = all_traj[current_valid_index]
        out['hist'], out['hist_mask'] = self.hist_process(all_traj)

        # gt
        # gt is nbrs ego-centric, which means prediction needed to be rotated in CaseVis
        out['gt'], out['gt_mask'] = self.gt_process(all_traj)
        # out['yaw'] = np.pad(all_traj[:, CURRENT, 5], [(0, MAX_AGENT_NUM - valid_agent_num)])
        out['centroid'] = np.pad(all_traj[:, CURRENT, :2], [(0, MAX_AGENT_NUM - valid_agent_num), (0, 0)])

        # extra info-------------------------------
        # lane info
        out['lane_vector'], lane_num = self.lane_process(data['lane'],data['traf_p_c_f'][CURRENT])

        # agents' adjacent lane index in the lane_vector, size=[agent_num,adj_lane_num]
        out['adj_index'], out['adj_mask'] = self.map_allocation(all_traj[..., CURRENT, :2], all_traj[..., CURRENT, 5],
                                                                out['lane_vector'][:lane_num])

        # trunk related----------------------#
        adj_lane_num = np.max(np.sum(out['adj_mask'], axis=1))
        out['valid_len'] = np.array([valid_agent_num, lane_num, adj_lane_num])
        out['max_len'] = np.array([MAX_LANE_NUM, MAX_NBRS_NUM])

        # predict list
        motion_list = data['tracks_to_predict'][current_valid_index]
        interaction_list = data['objects_of_interest'][current_valid_index]
        out['tracks_to_predict'] = np.pad(motion_list, [0, MAX_NBRS_NUM + 1 - valid_agent_num]).astype(bool)
        out['objects_of_interest'] = np.pad(interaction_list, [0, MAX_NBRS_NUM + 1 - valid_agent_num]).astype(bool)

        # obj type
        out['obj_type'] = np.pad(all_traj[:, CURRENT, 9], (0, MAX_NBRS_NUM + 1 - valid_agent_num)).astype(int)
        # out['velocity'] = np.pad(all_traj[..., 3:5], [(0, MAX_NBRS_NUM + 1 - valid_agent_num), (0, 0), (0, 0)])
        # out['agent_id'] = np.pad(all_traj[:, CURRENT, -1], (0, MAX_NBRS_NUM + 1 - valid_agent_num)).astype(int)
        try:
            out['state_id'] = str(data['id'][0], 'utf-8')
        except:
            pass

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='0')
    args = parser.parse_args()
    cfg = load_config_data(f"../config/{args.cfg}.yaml")
    dataset_cfg = cfg['dataset_cfg']
    dataset_cfg['cache'] = False
    dir = dataset_cfg['dataset_dir']
    cache_root = dir[:dir.find('trans')]

    periods = ['training', 'validation', 'testing', 'validation_interactive', 'testing_interactive']
    batch_size = 64

    for period in periods:
        ds = WaymoDataset(dataset_cfg, period)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
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
