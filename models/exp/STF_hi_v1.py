import torch.nn as nn
import torch
from models.utils import (
    Encoder, EncoderLayer,
    ChoiceHead,
    MultiHeadAttention, PointerwiseFeedforward,

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


class STF_hi_v1(STF):
    def __init__(self, cfg):
        super(STF_hi_v1, self).__init__(cfg)
        prop_num = cfg['prop_num']
        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        dec_out_size = cfg['out_dims']
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PointerwiseFeedforward(d_model, d_model * 2, dropout)
        pos_dim = 16
        # num of proposal
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.prediction_head = ChoiceHead(d_model * 2, dec_out_size, dropout)

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

    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        batch_size = valid_len.shape[0]
        # trajectory module
        hist = data['hist'][:, :max_agent]
        center = hist[...,-1,2:]
        yaw = data['misc'][:,:max_agent,10,4]
        yaw_1 = torch.cat([torch.cos(yaw).unsqueeze(-1),torch.sin(yaw).unsqueeze(-1)],-1)
        center_emb = self.cent_emb(center)
        yaw_emb = self.yaw_emb(yaw_1)

        hist[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[...,:2] = self._rotate(hist[...,:2],yaw)
        hist[...,2:4] = self._rotate(hist[...,2:4],yaw)

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)
        hist_out = torch.cat([center_emb.unsqueeze(dim=2).repeat(1, 1, self.query_batches.shape[-2], 1), hist_out], dim=-1)
        hist_out = self.fusion1cent(hist_out)
        hist_out = torch.cat([yaw_emb.unsqueeze(dim=2).repeat(1, 1, self.query_batches.shape[-2], 1), hist_out], dim=-1)
        hist_out = self.fusion1yaw(hist_out)

        # TODO: lane module
        # TODO: Traffic_light module
        # TODO: high-order interaction module
        social_valid_len = data['valid_len'][:, 1] + 1
        social_mask = torch.zeros((batch_size, 1, max_agent)).to(hist_out.device)
        for i in range(batch_size):
            social_mask[i, 0, :social_valid_len[i]] = 1

        social_inp = hist_out
        channel = social_inp.shape[-2]
        social_inp = torch.transpose(social_inp,1,2).reshape(-1,social_inp.shape[1],social_inp.shape[-1])
        social_mask = social_mask.repeat(1,channel,1).reshape(-1,1,social_mask.shape[-1])

        #social_emb = self.social_emb(social_inp.view(*hist_out.shape[:2], -1))
        social_mem = self.social_enc(social_inp, social_mask)
        #social_out = social_mem.unsqueeze(dim=2).repeat(1, 1, hist_out.shape[-2], 1)
        social_mem = social_mem.reshape(batch_size,channel,*social_mem.shape[1:])
        social_out = torch.transpose(social_mem,1,2)
        out = torch.cat([social_out, hist_out], -1)

        gather_list, new_data = self._gather_new_data(data, max_agent)

        # gather lane out
        gather_lane = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *out.shape[-2:])
        out = torch.gather(out, 1, gather_lane)

        outputs_coord, outputs_class = self.prediction_head(out, new_data['obj_type'])

        return outputs_coord, outputs_class, new_data
