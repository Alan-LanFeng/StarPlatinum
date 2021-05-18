import torch.nn as nn
import torch
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, LaneNet
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


class STF_traf(STF):
    def __init__(self, cfg):
        super(STF_traf, self).__init__(cfg)

        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']

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

    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        max_lane = torch.max(valid_len[:, 1])
        max_adj_lane = torch.max(valid_len[:, 2])
        batch_size = valid_len.shape[0]
        # trajectory module
        hist = data['hist'][:, :max_agent]

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # TODO: lane module
        lane_vector = torch.cat(
            [data['traffic_light'].view(*data['traffic_light'].shape, 1, 1).repeat(1, 1, 9, 1), data['lane_vector']],
            -1)
        lane_vector[data['lane_vector'][..., -1] == 0] = 0
        lane = lane_vector[:, :max_lane]
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

        # TODO: Traffic_light module
        # TODO: high-order interaction module

        gather_list, new_data = self._gather_new_data(data, max_agent)

        # gather lane out
        gather_lane = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *lane_out.shape[-2:])
        lane_out = torch.gather(lane_out, 1, gather_lane)

        outputs_coord, outputs_class = self.prediction_head(lane_out, new_data['obj_type'])

        return outputs_coord, outputs_class, new_data

    def _gather_new_data(self, data, max_agent):
        # select predict list and gather needed data
        tracks_to_predict = data['tracks_to_predict'][:, :max_agent]
        gather_list, gather_mask = bool2index(tracks_to_predict)
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
