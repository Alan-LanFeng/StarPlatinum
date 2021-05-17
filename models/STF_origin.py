import torch.nn as nn
import torch
from models.utils import (
    EncoderDecoder, Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward, SublayerConnection,
    LinearEmbedding, PositionalEncoding,
    GeneratorWithParallelHeads626, ChoiceHead, LaneNet
)
import sys
import copy


def bool2index(mask):
    seq = torch.arange(mask.shape[-1]).to(mask.device)
    index = mask * seq
    index[mask == 0] = 1000
    index = index.sort(-1).values.to(torch.int64)
    mask = index < 1000
    index[index == 1000] = 0
    return index, mask


class STF(nn.Module):
    def __init__(self, cfg):
        super(STF, self).__init__()

        # num of proposal
        prop_num = cfg['prop_num']
        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        traj_dims = cfg['traj_dims']
        dec_out_size = cfg['out_dims']

        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_model * 2, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.lane_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

        self.max_pred_num = cfg['max_pred_num']
        self.query_embed = nn.Embedding(prop_num, d_model)
        self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)

        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(traj_dims, d_model), c(position))
        )

        self.prediction_head = ChoiceHead(d_model * 2, dec_out_size, dropout)

        self.social_emb = nn.Sequential(
            nn.Linear(prop_num * d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_agent = torch.max(valid_len[:, 0])
        max_lane = torch.max(valid_len[:, 1])
        max_adj_lane = torch.max(valid_len[:, 2])
        batch_size = valid_len.shape[0]

        # =============trajectory module===========================
        hist = data['hist'][:, :max_agent]
        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # =============lane module===============================
        lane = data['lane_vector'][:, :max_lane]
        lane_enc = self.lanenet(lane)
        lane_valid_len = data['valid_len'][:, 0]
        lane_mask = torch.zeros((batch_size, 1, max_lane)).to(lane_enc.device)
        for i in range(batch_size):
            lane_mask[i, 0, :lane_valid_len[i]] = 1
        lane_mem = self.lane_enc(self.lane_emb(lane_enc), lane_mask)
        lane_mem = lane_mem.unsqueeze(1).repeat(1, max_agent, 1, 1)
        adj_index = data['adj_index'][:, :max_agent, :max_lane]
        adj_mask = data['adj_mask'][:, :max_agent, :max_lane]
        adj_index = adj_index.unsqueeze(-1).repeat(1, 1, 1, 128)[:, :, :max_adj_lane]
        adj_mask = adj_mask.unsqueeze(2)[:, :, :, :max_adj_lane]
        lane_mem = torch.gather(lane_mem, dim=-2, index=adj_index)
        lane_out = self.lane_dec(hist_out, lane_mem, adj_mask, None)

        # ===================interaction module=============================
        social_valid_len = data['valid_len'][:, 1] + 1
        social_mask = torch.zeros((batch_size, 1, max_agent)).to(lane_out.device)
        for i in range(batch_size):
            social_mask[i, 0, :social_valid_len[i]] = 1
        social_emb = self.social_emb(lane_out.view(*lane_out.shape[:2], -1))
        social_mem = self.social_enc(social_emb, social_mask)
        social_out = social_mem.unsqueeze(dim=2).repeat(1, 1, hist_out.shape[-2], 1)
        out = torch.cat([social_out, lane_out], -1)

        # TODO: Traffic_light module

        # select predict list and gather needed data
        tracks_to_predict = data['tracks_to_predict'][:, :max_agent]
        gather_list, gather_mask = bool2index(tracks_to_predict)
        gather_list, gather_mask = gather_list[:, :self.max_pred_num], gather_mask[:, :self.max_pred_num]
        # gather hist out
        gather_hist = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *out.shape[-2:])
        out = torch.gather(out, 1, gather_hist)
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
        cur_pos = torch.cat((data['yaw'][:, :max_agent].unsqueeze(-1), data['centroid'][:, :max_agent]), -1)
        gather_pos = gather_list.view(*gather_list.shape, 1).repeat(1, 1, cur_pos.shape[-1])
        cur_pos = torch.gather(cur_pos, 1, gather_pos)
        vel = data['velocity'][:, :max_agent]
        gather_vel = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *vel.shape[-2:])
        vel = torch.gather(vel, 1, gather_vel)
        agent_id = data['agent_id'][:, :max_agent]
        agent_id = torch.gather(agent_id, 1, gather_list)
        new_data = {
            'gt': gt,
            'gt_mask': gt_mask,
            'cur_pos': cur_pos,
            'vel': vel,
            'agent_id': agent_id,
            'tracks_to_predict': gather_mask
        }
        outputs_coord, outputs_class = self.prediction_head(out, obj_type)

        return outputs_coord, outputs_class, new_data
