import torch
import torch.nn as nn
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, LaneNet,PositionalEncoding,
)

import copy


def bool2index(mask):
    seq = torch.arange(mask.shape[-1]).to(mask.device)
    index = mask * seq
    index[mask == 0] = 1000
    index = index.sort(-1).values.to(torch.int64)
    mask = index < 1000
    index[index == 1000] = 0
    return index, mask


class road_roller(nn.Module):
    def __init__(self, cfg):
        super(road_roller, self).__init__()

        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        traj_dims = cfg['traj_dims']
        pos_dim = 16
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_model * 2, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.pos_emb = nn.Sequential(LinearEmbedding(traj_dims, d_model), c(position))
        self.traj_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        # num of proposal
        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])

        self.trajnet = LaneNet(
            cfg['traj_dims'],
            64,
            cfg['num_subgraph_layers'])


        self.lane_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

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
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

        self.prediction_head = nn.Sequential(
                        PointerwiseFeedforward(128, 2 * 128, dropout=dropout),
                        nn.Linear(128, 64, bias=True),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 32, bias=True),
                        nn.Linear(32, 1, bias=True),
                        nn.Sigmoid())

    def forward(self, disc,tracks_to_predict):
        lane_mask = disc['output_lane_mask']
        lane = disc['output_lane']
        # reconstruct each trajectory
        max_adj_lane = torch.max(lane_mask.squeeze(-2).sum(-1))
        lane = lane[:,:,:max_adj_lane]
        lane_mask = lane_mask[...,:max_adj_lane]
        batch,agent,prop = disc['traj'].shape[0],disc['traj'].shape[1],disc['traj'].shape[2]
        # ========traj enc=============
        hist_mask =disc['hist_mask']
        future_mask = disc['pred_mask']
        all_mask = torch.cat([hist_mask, future_mask], -1)
        traj = disc['traj']
        traj = traj.reshape(traj.shape[0],-1,*traj.shape[-2:])
        all_mask = all_mask.unsqueeze(2).repeat(1,1,prop,1,1).squeeze(-2).reshape(batch,-1,90)
        all_mask = all_mask.unsqueeze(-1).repeat(1,1,1,7)
        traj = traj*all_mask
        traj_emb = self.trajnet(traj)
        traj_emb = traj_emb.reshape(batch,agent,prop,-1)

        # =========road graph enc
        lane = lane.reshape(lane.shape[0], -1, *lane.shape[-2:])
        lane_enc = self.lanenet(lane)
        lane_enc = lane_enc.reshape(lane_enc.shape[0], agent, max_adj_lane, lane_enc.shape[-1])
        lane_enc = lane_enc.reshape(-1, *lane_enc.shape[-2:])

        lane_mask = lane_mask.reshape(-1,*lane_mask.shape[-2:])
        lane_mem = self.lane_enc(self.lane_emb(lane_enc), lane_mask)
        lane_mask = lane_mask.reshape(lane.shape[0], -1, *lane_mask.shape[-2:])
        lane_mem = lane_mem.reshape(lane.shape[0],-1, *lane_mem.shape[-2:])
        lane_out = self.lane_dec(traj_emb, lane_mem, lane_mask, None)

        # =============social model
        # road roller da
        out = self.prediction_head(lane_out)
        return out


    def _rotate(self, vec, yaw):
        c, s = torch.cos(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2]), torch.sin(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2])
        vec[..., 0], vec[..., 1] = c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]
        return vec
