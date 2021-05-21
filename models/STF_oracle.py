import torch
from .STF import STF


class STF_oracle(STF):
    def __init__(self, cfg, device):
        super(STF_oracle, self).__init__(cfg, device)

    def forward(self, data):
        valid_len = data['valid_len']
        self.device = valid_len.get_device()
        max_len = data['max_len']
        max_agent = max(torch.max(valid_len[:, 0]),self.max_pred_num)
        # trajectory module
        hist = data['hist'][:, :max_agent]
        batch_size, _, horizon, channels = hist.shape
        ego_gt = data['misc'][:, 0, 10:, :2]  # batch, 91, 2
        ego_gt_mask = data['misc'][:, 0, 10:, -2]
        ego_gt_vec = torch.cat([ego_gt[..., :-1, :], ego_gt[..., 1:, :]], dim=-1).unsqueeze(1)
        ego_gt_vec = torch.cat([ego_gt_vec, torch.zeros(batch_size, max_agent-1, 80, channels).to(self.device)], dim=1)
        hist = torch.cat([hist, ego_gt_vec], dim=2)
        ego_gt_mask_vec = (ego_gt_mask[..., :-1] * ego_gt_mask[..., 1:]).unsqueeze(1).type(torch.bool)
        mask = torch.cat([ego_gt_mask_vec, torch.zeros([batch_size, max_agent-1, 80], dtype=torch.bool).to(self.device)], dim=1)
        hist_mask = data['hist_mask'].unsqueeze(-2)[:, :max_agent]
        hist_mask = torch.cat([hist_mask, mask.unsqueeze(-2)], dim=-1)
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
