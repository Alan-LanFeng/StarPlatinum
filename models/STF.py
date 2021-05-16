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


class STF(nn.Module):
    def __init__(self, cfg):
        super(STF, self).__init__()
        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
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

        # trajectory module
        hist = data['hist']
        hist_mask = data['hist_mask'].unsqueeze(-2)
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # TODO: lane module
        # TODO: Traffic_light module
        # TODO: high-order interaction module
        obj_type = data['obj_type'].to(torch.int64)
        outputs_coord, outputs_class = self.prediction_head(hist_out,obj_type)

        return outputs_coord, outputs_class
