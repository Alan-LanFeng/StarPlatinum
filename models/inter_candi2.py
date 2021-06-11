import torch
import torch.nn as nn
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, LaneNet,
    intChoiceHead
)
from models.STF import STF
import copy


def bool2index(mask):
    seq = torch.arange(mask.shape[-1]).to(mask.device)
    index = mask * seq
    index[mask == 0] = 1000
    index = index.sort(-1).values.to(torch.int64)
    mask = index < 1000
    index[index == 1000] = 0
    return index, mask


class inter_candi2(STF):
    def __init__(self, cfg):
        super(inter_candi2, self).__init__(cfg)

        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        prop_num = cfg['prop_num']
        dec_out_size = cfg['out_dims']
        pos_dim = 16
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_model * 2, dropout)
        # num of proposal
        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.lane_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.prediction_head = intChoiceHead(d_model * 2, dec_out_size, dropout)

        self.cent_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))
        self.yaw_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))

        self.fusion1cent = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.fusion1yaw = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))

        self.social_emb = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))

        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 2*N)
        self.social_fuse = nn.Sequential(
            nn.Linear(2*d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.adj_net = LaneNet(
            7,
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.tra_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)

    def forward(self, data: dict):

        valid_len = data['valid_len']
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        gather_list, new_data = self._gather_new_data(data, max_agent)
        # ============================one hot=======================
        obj_type = data['obj_type'][:,:max_agent].clone()
        obj_type-=1
        obj_type[obj_type<0]=0
        one = nn.functional.one_hot(obj_type,3).to(torch.float32)
        one_hot_hist = one.unsqueeze(-2).repeat(1,1,10,1)
        # =======================================trajectory module===================================
        hist_mask = data['hist_mask'][:, :max_agent]
        hist = data['hist'][:, :max_agent]
        center = data['hist'][:, :max_agent][...,-1,2:].detach().clone()
        yaw = data['misc'][:,:max_agent,10,4].detach().clone()
        # =============calculate distance between agents============
        center_x = center[...,0]
        center_y = center[...,1]
        center_x1 = center_x.unsqueeze(-1).repeat(1,1,center_x.shape[1])
        center_x2 = center_x.unsqueeze(-2).repeat(1,center_x.shape[1],1)
        dist_x = center_x1-center_x2
        center_y1 = center_y.unsqueeze(-1).repeat(1,1,center_y.shape[1])
        center_y2 = center_y.unsqueeze(-2).repeat(1,center_y.shape[1],1)
        dist_y = center_y1-center_y2
        dist = torch.sqrt(torch.square(dist_x)+torch.square(dist_y))
        perception_mask = dist < 50
        social_valid_len = data['valid_len'][:, 0] + 1
        for i in range(perception_mask.shape[0]):
            perception_mask[i, social_valid_len[i]:, :] = False
            perception_mask[i, :, social_valid_len[i]:] = False
        gather_ = gather_list.view(*gather_list.shape,1).repeat(1,1,perception_mask.shape[-1])
        perception_mask = torch.gather(perception_mask,1,gather_)
        # =============================================================================
        adj_traj = hist.detach().clone()
        adj_traj = torch.cat([adj_traj,one_hot_hist],-1)
        adj_traj = adj_traj.unsqueeze(1).repeat(1,max_agent,1,1,1)
        adj_traj[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1,1).repeat(1,1,*adj_traj.shape[2:4],2)
        adj_traj[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2], 1, 1, 1).repeat(1, 1, *adj_traj.shape[2:4], 2)
        adj_traj = adj_traj.reshape(*adj_traj.shape[:2],-1,adj_traj.shape[-1])
        adj_traj[...,:2] = self._rotate(adj_traj[...,:2],yaw)
        adj_traj[..., 2:4] = self._rotate(adj_traj[..., 2:4], yaw)
        adj_traj = adj_traj.reshape(*adj_traj.shape[:2], -1,10, adj_traj.shape[-1])
        mask = hist_mask.unsqueeze(1).repeat(1,hist_mask.shape[-2],1,1).unsqueeze(-1).repeat(1,1,1,1,adj_traj.shape[-1])
        adj_traj = adj_traj*mask
        gather_ = gather_list.view(*gather_list.shape,1,1,1).repeat(1,1,*adj_traj.shape[-3:])
        adj_traj = torch.gather(adj_traj,1,gather_)
        adj_traj = adj_traj.reshape(adj_traj.shape[0],-1,*adj_traj.shape[-2:])
        traj_enc = self.adj_net(adj_traj)
        traj_enc = traj_enc.reshape(traj_enc.shape[0], 2,-1, traj_enc.shape[-1])
        traj_enc = traj_enc.reshape(-1, *traj_enc.shape[-2:])
        traj_enc = self.tra_emb(traj_enc)

        gather_adj_mask = gather_list.view(*gather_list.shape, 1).repeat(1, 1, 660)
        adj_mask = torch.gather(data['adj_mask'], 1, gather_adj_mask)
        max_adj_lane = torch.max(adj_mask.sum(-1))
        adj_index = torch.gather(data['adj_index'], 1, gather_adj_mask)[..., :max_adj_lane]
        adj_mask = adj_mask[..., :max_adj_lane]
        gather_hist = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, 10, 4)
        hist = torch.gather(data['hist'], 1, gather_hist)
        hist_mask = data['hist_mask']
        hist_mask = (torch.gather(hist_mask, 1, gather_list.unsqueeze(-1).repeat(1, 1, 10))).unsqueeze(-2)
        center = hist[:, :, -1, 2:].detach().clone()
        yaw = new_data['misc'][:, :, 10, 4].detach().clone()
        obj_type = new_data['obj_type'].detach().clone()
        obj_type-=1
        obj_type[obj_type<0]=0
        one = nn.functional.one_hot(obj_type,3).to(torch.float32)
        one_hot_hist = one.unsqueeze(-2).repeat(1,1,10,1)
        hist[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[...,:2] = self._rotate(hist[...,:2],yaw)
        hist[...,2:4] = self._rotate(hist[...,2:4],yaw)
        hist = torch.cat([hist,one_hot_hist],-1)

        #hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)
        # ====================================rg and ts module========================================
        lane = data['lane_vector']
        lane = lane.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        adj_index = adj_index.view(*adj_index.shape,1,1).repeat(1,1,1,*lane.shape[-2:])
        adj_mask = adj_mask.unsqueeze(2)
        lane = torch.gather(lane,2,adj_index)

        lane[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1,1).repeat(1,1,*lane.shape[2:4],2)
        lane[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2], 1, 1, 1).repeat(1, 1, *lane.shape[2:4], 2)
        lane = lane.reshape(*lane.shape[:2],-1,lane.shape[-1])
        lane[...,:2] = self._rotate(lane[...,:2],yaw)
        lane[..., 2:4] = self._rotate(lane[..., 2:4], yaw)
        lane = lane.reshape(*lane.shape[:2], -1,9, lane.shape[-1])
        lane = lane.reshape(lane.shape[0],-1,*lane.shape[-2:])

        lane_enc = self.lanenet(lane)
        lane_enc = lane_enc.reshape(lane_enc.shape[0],2,max_adj_lane,lane_enc.shape[-1])
        lane_enc = lane_enc.reshape(-1,*lane_enc.shape[-2:])
        lane_mask = adj_mask.reshape(-1,*adj_mask.shape[-2:])
        lane_enc = self.lane_emb(lane_enc)
        lane_enc = torch.cat([lane_enc,traj_enc],-2)
        perception_mask = perception_mask.reshape(-1,perception_mask.shape[-1]).unsqueeze(1)
        lane_mask = torch.cat([lane_mask, perception_mask], -1)
        lane_mem = self.lane_enc(lane_enc, lane_mask)
        lane_mem = lane_mem.reshape(*hist_out.shape[:2],*lane_mem.shape[-2:])
        lane_mem = lane_mem.reshape(*hist_out.shape[:2],*lane_mem.shape[-2:])
        perception_mask = perception_mask.reshape(*adj_mask.shape[:2],*perception_mask.shape[-2:])
        adj_mask = torch.cat([adj_mask,perception_mask],-1)
        lane_out = self.lane_dec(hist_out, lane_mem, adj_mask, None)

        # ===================high-order interaction module=============================================
        yaw_1 = torch.cat([torch.cos(yaw).unsqueeze(-1),torch.sin(yaw).unsqueeze(-1)],-1)
        center_emb = self.cent_emb(center)
        yaw_emb = self.yaw_emb(yaw_1)
        social_emb = self.social_emb(lane_out)
        center_emb = center_emb.unsqueeze(-2).repeat(1,1,social_emb.shape[-2],1)
        yaw_emb = yaw_emb.unsqueeze(-2).repeat(1,1,social_emb.shape[-2],1)
        social_emb = torch.cat([center_emb, social_emb], dim=-1)
        social_emb = self.fusion1cent(social_emb)
        social_emb = torch.cat([yaw_emb, social_emb], dim=-1)
        social_emb = self.fusion1yaw(social_emb)

        social_emb = social_emb.transpose(1,2).reshape(-1,2,128)

        social_mem = self.social_enc(social_emb, None)
        social_mem = social_mem.reshape(lane.shape[0],-1,*social_mem.shape[-2:])
        social_out = social_mem.transpose(1,2)
        out = torch.cat([social_out, lane_out], -1)
        out = self.social_fuse(out)


        outputs_coord, outputs_class = self.prediction_head(out, new_data['obj_type'])

        return outputs_coord, outputs_class, new_data

    def _gather_new_data(self, data, max_agent):
        # select predict list and gather needed data
        tracks_to_predict = data['objects_of_interest'][:, :max_agent]
        gather_list, gather_mask = self._bool2index(tracks_to_predict)
        gather_list, gather_mask = gather_list[:, :self.max_pred_num], gather_mask[:, :self.max_pred_num]
        # gather obj type
        obj_type = data['obj_type'].to(torch.int64)[:, :max_agent]
        obj_type = torch.gather(obj_type, 1, gather_list)

        # gather other data
        gt = data['gt'][:, :max_agent]
        gather_gt = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *gt.shape[-2:])
        gt = torch.gather(gt, 1, gather_gt)
        gt_mask = data['gt_mask'][:, :max_agent]
        gather_gt_mask = gather_list.view(*gather_list.shape, 1).repeat(1, 1, gt_mask.shape[-1])
        gt_mask = torch.gather(gt_mask, 1, gather_gt_mask)

        misc = data['misc'][:, :max_agent]
        misc = torch.gather(misc, 1, gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *misc.shape[-2:]))

        centroid = data['centroid'][:, :max_agent]
        gather_centroid = gather_list.unsqueeze(-1).repeat(1, 1, 2)
        centroid = torch.gather(centroid, 1, gather_centroid)

        new_data = {
            'gt': gt,
            'gt_mask': gt_mask,
            'tracks_to_predict': gather_mask,
            'misc': misc,
            'obj_type': obj_type,
            'centroid': centroid
        }
        return gather_list, new_data
