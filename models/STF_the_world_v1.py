import torch
import torch.nn as nn
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, LaneNet,
    ChoiceHead
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


class STF_the_world_v1(STF):
    def __init__(self, cfg):
        super(STF_the_world_v1, self).__init__(cfg)

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

        self.social_emb = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.social_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)


    def forward(self, data: dict):
        valid_len = data['valid_len']
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        max_lane = max(torch.max(valid_len[:, 1]), 1)
        max_adj_lane = max(torch.max(valid_len[:, 2]), 1)
        disc = {}

        # ============================one hot=======================
        obj_type = data['obj_type'][:,:max_agent].clone()
        obj_type-=1
        obj_type[obj_type<0]=0
        one = nn.functional.one_hot(obj_type,3).to(torch.float32)
        one_hot_hist = one.unsqueeze(-2).repeat(1,1,10,1)
        # =======================================trajectory module===================================
        hist = data['hist'][:, :max_agent]
        center = data['hist'][:, :max_agent][...,-1,2:].detach().clone()
        yaw = data['misc'][:,:max_agent,10,4].detach().clone()

        yaw_1 = torch.cat([torch.cos(yaw).unsqueeze(-1),torch.sin(yaw).unsqueeze(-1)],-1)
        center_emb = self.cent_emb(center)
        yaw_emb = self.yaw_emb(yaw_1)

        hist[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[...,:2] = self._rotate(hist[...,:2],yaw)
        hist[...,2:4] = self._rotate(hist[...,2:4],yaw)

        hist = torch.cat([hist, one_hot_hist], -1)

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # ====================================rg and ts module========================================
        lane = data['lane_vector'][:, :max_lane]
        lane = lane.unsqueeze(1).repeat(1, max_agent, 1, 1, 1)

        adj_index = data['adj_index'][:, :max_agent, :max_lane]
        adj_mask = data['adj_mask'][:, :max_agent, :max_lane]
        adj_index = adj_index.reshape(*adj_index.shape,1,1).repeat(1, 1, 1,*lane.shape[-2:])[:, :, :max_adj_lane]
        adj_mask = adj_mask.unsqueeze(2)[:, :, :, :max_adj_lane]
        lane = torch.gather(lane,2,adj_index)

        lane[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1,1).repeat(1,1,*lane.shape[2:4],2)
        lane[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2], 1, 1, 1).repeat(1, 1, *lane.shape[2:4], 2)
        lane = lane.reshape(*lane.shape[:2],-1,lane.shape[-1])
        lane[...,:2] = self._rotate(lane[...,:2],yaw)
        lane[..., 2:4] = self._rotate(lane[..., 2:4], yaw)
        lane = lane.reshape(*lane.shape[:2], -1,9, lane.shape[-1])
        # =====================disc==========================================
        output_lane = lane
        # =====================disc==========================================
        lane = lane.reshape(lane.shape[0],-1,*lane.shape[-2:])
        lane_enc = self.lanenet(lane)
        lane_enc = lane_enc.reshape(lane_enc.shape[0],max_agent,max_adj_lane,lane_enc.shape[-1])
        lane_enc = lane_enc.reshape(-1,*lane_enc.shape[-2:])
        lane_mask = adj_mask.reshape(-1,*adj_mask.shape[-2:])

        lane_mem = self.lane_enc(self.lane_emb(lane_enc), lane_mask)
        lane_mem = lane_mem.reshape(*hist_out.shape[:2],*lane_mem.shape[-2:])

        lane_out = self.lane_dec(hist_out, lane_mem, adj_mask, None)

        # ===================high-order interaction module=============================================
        social_valid_len = data['valid_len'][:, 0] + 1
        social_mask = torch.zeros((lane_out.shape[0], 1, max_agent)).to(lane_out.device)
        for i in range(lane_out.shape[0]):
            social_mask[i, 0, :social_valid_len[i]] = 1
        social_emb = self.social_emb(lane_out)
        social_emb = torch.max(social_emb, -2)[0]
        social_emb = torch.cat([center_emb, social_emb], dim=-1)
        social_emb = self.fusion1cent(social_emb)
        social_emb = torch.cat([yaw_emb, social_emb], dim=-1)
        social_emb = self.fusion1yaw(social_emb)

        social_mem = self.social_enc(social_emb, social_mask)
        social_out = social_mem.unsqueeze(dim=2).repeat(1, 1, hist_out.shape[-2], 1)
        out = torch.cat([social_out, lane_out], -1)

        # gather
        gather_list, new_data = self._gather_new_data(data, max_agent)

        # ===================gather road graph for discriminator====
        gather_road = gather_list.view(*gather_list.shape,1,1,1).repeat(1,1,*output_lane.shape[-3:])
        output_lane = torch.gather(output_lane,1,gather_road)
        output_lane = nn.functional.pad(output_lane,[0,0,0,0,0,660-output_lane.shape[-3]])
        gather_mask = gather_list.view(*gather_list.shape,1,1).repeat(1,1,*adj_mask.shape[-2:])
        output_lane_mask = torch.gather(adj_mask,1,gather_mask)
        output_lane_mask = nn.functional.pad(output_lane_mask,[0,660-output_lane_mask.shape[-1]])

        disc['output_lane'] = output_lane
        disc['output_lane_mask'] = output_lane_mask
        # ===========================================================

        # Star Platinum, the world!
        gather_out = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *out.shape[-2:])
        out = torch.gather(out, 1, gather_out)
        outputs_coord, outputs_class = self.prediction_head(out, new_data['obj_type'])


        return outputs_coord, outputs_class, new_data,{}
