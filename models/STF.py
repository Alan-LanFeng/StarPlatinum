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
        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.max_pred_num = cfg['max_pred_num']

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

        self.query_embed = nn.Embedding(prop_num, d_model)
        self.query_embed.weight.requires_grad == False
        nn.init.orthogonal_(self.query_embed.weight)

        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(traj_dims, d_model), c(position))
        )
        self.prediction_head = ChoiceHead(d_model, dec_out_size, dropout)

    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_len = data['max_len']
        max_agent = max(torch.max(valid_len[:, 0]),self.max_pred_num)
        # trajectory module
        hist = data['hist'][:, :max_agent]

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # TODO: lane module
        # TODO: Traffic_light module
        # TODO: high-order interaction module

        # select predict list and gather needed data
        tracks_to_predict = data['tracks_to_predict'][:, :max_agent]
        gather_list, gather_mask = bool2index(tracks_to_predict)
        gather_list, gather_mask = gather_list[:, :self.max_pred_num], gather_mask[:, :self.max_pred_num]
        # gather hist out
        gather_hist = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *hist_out.shape[-2:])
        hist_out = torch.gather(hist_out, 1, gather_hist)
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

        misc = data['misc'][:, :max_agent]
        misc = torch.gather(misc, 1, gather_list.view(*gather_list.shape, 1, 1).repeat(1,1,*misc.shape[-2:]))

        centroid = data['centroid'][:, :max_agent]
        gather_centroid = gather_list.unsqueeze(-1).repeat(1, 1, 2)
        centroid = torch.gather(centroid, 1, gather_centroid)

        new_data = {
            'gt': gt,
            'gt_mask': gt_mask,
            'cur_pos': cur_pos,
            'vel': vel,
            'agent_id': agent_id,
            'tracks_to_predict': gather_mask,
            'misc': misc,
            'obj_type': obj_type,
            'centroid': centroid
        }

        outputs_coord, outputs_class = self.prediction_head(hist_out, obj_type)

        return outputs_coord, outputs_class, new_data
