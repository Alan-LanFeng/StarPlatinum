import torch.nn as nn
import torch
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding,
    ChoiceHead, LaneNet
)
from models.STF import STF
import copy


class STF_rg_hi_pf(STF):
    def __init__(self, cfg):
        super(STF_rg_hi_pf, self).__init__(cfg)

        # num of proposal
        prop_num = cfg['prop_num']
        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        dec_out_size = cfg['out_dims']
        pos_dim = 64
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_model * 2, dropout)

        self.lanenet = LaneNet(
            cfg['lane_dims'],
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.lane_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

        self.max_pred_num = cfg['max_pred_num']

        self.prediction_head = ChoiceHead(d_model * 2, dec_out_size, dropout)

        self.social_emb = nn.Sequential(
            nn.Linear(prop_num * d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))
        self.fusion1 = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.fusion2 = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))

    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        max_lane = torch.max(valid_len[:, 1])
        max_adj_lane = torch.max(valid_len[:, 2])
        batch_size = valid_len.shape[0]

        # =============trajectory module===========================
        hist = data['hist'][:, :max_agent]
        center = hist[...,-1,2:]
        #yaw = data['misc'][:,:max_agent,10,4]
        #pos = torch.cat([center,yaw.unsqueeze(-1)],-1)

        hist[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        # hist[...,:2] = self._rotate(hist[...,:2],yaw)
        # hist[...,2:4] = self._rotate(hist[...,2:4],yaw)

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        pos = self.pos_emb(center)
        hist_out = torch.cat([pos.unsqueeze(dim=2).repeat(1, 1, self.query_batches.shape[-2], 1), hist_out], dim=-1)
        hist_out = self.fusion1(hist_out)


        # rg module
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
        social_emb = self.fusion2(torch.cat([pos, social_emb], -1))
        social_mem = self.social_enc(social_emb, social_mask)
        social_out = social_mem.unsqueeze(dim=2).repeat(1, 1, hist_out.shape[-2], 1)
        out = torch.cat([social_out, lane_out], -1)

        # TODO: Traffic_light module

        # select predict list and gather needed data
        gather_list, new_data = self._gather_new_data(data, max_agent)

        # gather hist out
        gather_out = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *out.shape[-2:])
        out = torch.gather(out, 1, gather_out)

        outputs_coord, outputs_class = self.prediction_head(out, new_data['obj_type'])

        return outputs_coord, outputs_class, new_data
    def _rotate(self, vec, yaw):
        c, s = torch.cos(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2]), torch.sin(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2])
        vec[..., 0], vec[..., 1] = c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]
        return vec
