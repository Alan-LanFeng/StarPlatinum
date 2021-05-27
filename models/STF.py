import torch.nn as nn
import torch
from models.utils import (
    EncoderDecoder, Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    MultiHeadAttention, PointerwiseFeedforward,
    LinearEmbedding, PositionalEncoding,
    ChoiceHead
)
import copy


class STF(nn.Module):
    def __init__(self, cfg):
        super(STF, self).__init__()
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
        max_agent = max(torch.max(valid_len[:, 0]), self.max_pred_num)
        # trajectory module
        hist = data['hist'][:, :max_agent]

        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*hist.shape[:2],
                                                                                                       1, 1)
        hist_out = self.hist_tf(hist, self.query_batches, hist_mask, None)

        # TODO: lane module
        # TODO: Traffic_light module
        # TODO: high-order interaction module

        gather_list, new_data = self._gather_new_data(data, max_agent)

        # gather hist out
        gather_hist = gather_list.view(*gather_list.shape, 1, 1).repeat(1, 1, *hist_out.shape[-2:])
        hist_out = torch.gather(hist_out, 1, gather_hist)

        outputs_coord, outputs_class = self.prediction_head(hist_out, new_data['obj_type'])

        return outputs_coord, outputs_class, new_data

    def _exp_hist_with_ego_gt(self, data: dict, hist: torch.Tensor, hist_mask: torch.Tensor,
                             center: torch.Tensor, yaw: torch.Tensor):
        batch_size, car_num, horizon, channels = hist.shape
        self.device = hist.get_device()
        exp_hist = torch.zeros([batch_size, car_num, 90, channels]).to(self.device)
        exp_hist_mask = torch.zeros([batch_size, car_num, 1, 90]).to(self.device)
        exp_hist[:, :, :10, :] = hist.detach().clone()
        exp_hist_mask[:, :, :, :10] = hist_mask.detach().clone()
        for i in range(data['misc'].shape[0]):
            idx = torch.where(data['tracks_to_predict'][i] == True)[0][0]
            ego_gt = data['misc'][i, idx, 10:, :2].detach().clone()# batch, 91, 2
            ego_gt[:, 0] -= center[i, idx, 0]
            ego_gt[:, 1] -= center[i, idx, 1]
            ego_gt[:, :2] = self._rotate(ego_gt[:, :2], yaw[i, idx])
            ego_gt_mask = data['misc'][i, idx, 10:, -2]
            exp_hist[i, idx, 10:, :] = torch.cat([ego_gt[:-1, :], ego_gt[1:, :]], dim=-1)
            exp_hist_mask[i, idx, :, 10:] = (ego_gt_mask[:-1] * ego_gt_mask[1:]).type(torch.bool)
        return exp_hist, exp_hist_mask

    def _gather_new_data(self, data, max_agent):
        # select predict list and gather needed data
        tracks_to_predict = data['tracks_to_predict'][:, :max_agent]
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

    def _bool2index(self, mask):
        seq = torch.arange(mask.shape[-1]).to(mask.device)
        index = mask * seq
        index[mask == 0] = 1000
        index = index.sort(-1).values.to(torch.int64)
        mask = index < 1000
        index[index == 1000] = 0
        return index, mask

    def _rotate(self, vec, yaw):
        c, s = torch.cos(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2]), torch.sin(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2])
        vec[..., 0], vec[..., 1] = c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]
        return vec
