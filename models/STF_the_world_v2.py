import torch
import torch.nn as nn
from models.utils import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, LaneNet,PositionalEncoding,
    ChoiceHead,EncoderDecoder,TypeEmb
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


class STF_the_world_v2(STF):
    def __init__(self, cfg):
        super(STF_the_world_v2, self).__init__(cfg)

        d_model = cfg['d_model']
        h = cfg['attention_head']
        dropout = cfg['dropout']
        N = cfg['model_layers_num']
        dec_out_size = cfg['out_dims']
        traj_dims = cfg['traj_dims']
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
        self.tra_emb = LinearEmbedding(cfg['subgraph_width_unit'] * 2, d_model)
        self.lane_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.lane_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.prediction_head = ChoiceHead(d_model, dec_out_size, dropout)
        self.type_lut = nn.Linear(3, 128)
        position = PositionalEncoding(d_model, dropout)
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
        self.adj_net = LaneNet(
            7,
            cfg['subgraph_width_unit'],
            cfg['num_subgraph_layers'])
        self.traj_enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.traj_emb = LinearEmbedding(d_model, d_model)
        self.traj_dec = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.traj_mlp = nn.Sequential(
            nn.Linear(192, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.hist_emb = LinearEmbedding(traj_dims, d_model)

        self.hist_type = TypeEmb(3,128)
        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(c(position),self.hist_type)
        )
        self.traj_type = TypeEmb(10,128)

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
        self.hist_type.add_type(one_hot_hist)
        # =======================================trajectory module===================================
        hist_mask = data['hist_mask'][:, :max_agent]
        hist = data['hist'][:, :max_agent]
        center = data['hist'][:, :max_agent][...,-1,2:].detach().clone()
        yaw = data['misc'][:,:max_agent,10,4].detach().clone()

        # =============calculate distance between agents============
        # center_x = center[...,0]
        # center_y = center[...,1]
        # center_x1 = center_x.unsqueeze(-1).repeat(1,1,center_x.shape[1])
        # center_x2 = center_x.unsqueeze(-2).repeat(1,center_x.shape[1],1)
        # dist_x = center_x1-center_x2
        # center_y1 = center_y.unsqueeze(-1).repeat(1,1,center_y.shape[1])
        # center_y2 = center_y.unsqueeze(-2).repeat(1,center_y.shape[1],1)
        # dist_y = center_y1-center_y2
        # dist = torch.sqrt(torch.square(dist_x)+torch.square(dist_y))
        # a = dist.numpy()
        # ========================================================

        yaw_1 = torch.cat([torch.cos(yaw).unsqueeze(-1),torch.sin(yaw).unsqueeze(-1)],-1)
        # center_emb = self.cent_emb(center)
        # yaw_emb = self.yaw_emb(yaw_1)
        #hist = torch.cat([hist, one_hot_hist], -1)
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

        # ===============================================================
        hist[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2],1,1).repeat(1,1,10,2)
        hist[...,:2] = self._rotate(hist[...,:2],yaw)
        hist[...,2:4] = self._rotate(hist[...,2:4],yaw)

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_emb = self.hist_emb(hist)
        hist_out = self.hist_tf(hist_emb, self.query_batches, hist_mask, None)

        # ====================================rg and ts module========================================
        lane = data['lane_vector'][:, :max_lane]
        lane = lane.unsqueeze(1).repeat(1, max_agent, 1, 1, 1)

        adj_index = data['adj_index'][:, :max_agent, :max_lane]
        adj_mask = data['adj_mask'][:, :max_agent, :max_lane]
        adj_index = adj_index.reshape(*adj_index.shape,1,1).repeat(1, 1, 1,*lane.shape[-2:])[:, :, :max_adj_lane]
        adj_mask = adj_mask.unsqueeze(2)[:, :, :, :max_adj_lane]
        lane = torch.gather(lane,2,adj_index)
        lane_type = lane[...,0,4:]
        lane[...,[0,2]]-=center[...,0].reshape(*center.shape[:2],1,1,1).repeat(1,1,*lane.shape[2:4],2)
        lane[..., [1, 3]] -= center[..., 1].reshape(*center.shape[:2], 1, 1, 1).repeat(1, 1, *lane.shape[2:4], 2)
        lane = lane.reshape(*lane.shape[:2],-1,lane.shape[-1])
        lane[...,:2] = self._rotate(lane[...,:2],yaw)
        lane[..., 2:4] = self._rotate(lane[..., 2:4], yaw)
        lane = lane.reshape(*lane.shape[:2], -1,9, lane.shape[-1])
        # =====================disc==========================================
        output_lane = lane
        # =====================disc==========================================
        social_valid_len = data['valid_len'][:, 0] + 1
        social_mask = torch.zeros((lane.shape[0], 1, max_agent)).to(lane.device)
        for i in range(adj_traj.shape[0]):
            social_mask[i, 0, :social_valid_len[i]] = 1
        social_mask = social_mask.repeat(1, social_mask.shape[-1], 1).unsqueeze(-2).reshape(-1, *social_mask.shape[-2:]).to(bool)
        adj_traj = adj_traj.reshape(adj_traj.shape[0],-1,*adj_traj.shape[-2:])
        traj_enc = self.adj_net(adj_traj)
        traj_enc = traj_enc.reshape(traj_enc.shape[0], max_agent, -1, traj_enc.shape[-1])
        traj_enc = traj_enc.reshape(-1, *traj_enc.shape[-2:])


        lane = lane.reshape(lane.shape[0],-1,*lane.shape[-2:])
        lane_enc = self.lanenet(lane)
        lane_enc = lane_enc.reshape(lane_enc.shape[0],max_agent,max_adj_lane,lane_enc.shape[-1])
        lane_enc = lane_enc.reshape(-1,*lane_enc.shape[-2:])
        lane_mask = adj_mask.reshape(-1,*adj_mask.shape[-2:])

        lane_mask = torch.cat([lane_mask,social_mask],-1)

        lane_enc = self.lane_emb(lane_enc)
        traj_enc = self.tra_emb(traj_enc)
        lane_enc = torch.cat([lane_enc,traj_enc],-2)

        one = one.unsqueeze(1).repeat(1,one.shape[-2],1,1)
        a = torch.zeros([*lane_type.shape[:3],3]).to(lane_type.device)
        lane_type = torch.cat([lane_type,a],-1)
        a = torch.zeros([*one.shape[:3],7]).to(lane_type.device)
        one = torch.cat([a,one],-1)
        all_type = torch.cat([lane_type,one],-2)
        all_type = all_type.reshape(-1,*all_type.shape[-2:])
        self.traj_type.add_type(all_type)
        lane_mem = self.lane_enc(lane_enc, lane_mask,self.traj_type)
        lane_mem = lane_mem.reshape(*hist_out.shape[:2],*lane_mem.shape[-2:])
        social_mask = social_mask.reshape(hist_out.shape[0],-1,*social_mask.shape[-2:])
        adj_mask = torch.cat([adj_mask,social_mask],-1)
        lane_out = self.lane_dec(hist_out, lane_mem, adj_mask, None)

        # ===================high-order interaction module=============================================
        # adj_traj = adj_traj.reshape(adj_traj.shape[0],-1,*adj_traj.shape[-2:])
        # traj_enc = self.adj_net(adj_traj)
        # traj_enc = traj_enc.reshape(traj_enc.shape[0],max_agent,max_agent,lane_enc.shape[-1])
        #
        # social_emb = self.social_emb(lane_out)
        # social_emb = torch.max(social_emb, -2)[0].unsqueeze(1).repeat(1,max_agent,1,1)
        # traj_enc = torch.cat([traj_enc,social_emb],-1)
        # traj_enc = self.traj_mlp(traj_enc)
        #
        # traj_enc = traj_enc.reshape(-1,*traj_enc.shape[-2:])
        # social_valid_len = data['valid_len'][:, 1] + 1
        # social_mask = torch.zeros((lane_out.shape[0], 1, max_agent)).to(lane_out.device)
        # for i in range(lane_out.shape[0]):
        #     social_mask[i, 0, :social_valid_len[i]] = 1
        # social_mask = social_mask.repeat(1,social_mask.shape[-1],1).unsqueeze(-2).reshape(-1,*social_mask.shape[-2:])
        # traj_mem = self.traj_enc(traj_enc, social_mask)
        # traj_mem = traj_mem.reshape(*lane_out.shape[:2],*traj_mem.shape[-2:])
        # social_mask = social_mask.reshape(*lane_out.shape[:2],*social_mask.shape[-2:])
        # out = self.traj_dec(lane_out, traj_mem, social_mask, None)

        out = lane_out
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

        return outputs_coord, outputs_class, new_data,disc
