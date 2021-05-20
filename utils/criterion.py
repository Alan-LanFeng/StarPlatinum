import torch

eps = 1e-9


class Loss(torch.nn.Module):

    def __init__(self, cfg, device):
        super(Loss, self).__init__()
        self.alpha = 0.01
        self.K = cfg['K']

    def forward(self, pred):
        pred, conf, data = pred[0], pred[1], pred[2]
        gt, gt_mask = data['gt'], data['gt_mask']
        gt_valid_len = torch.sum(gt_mask, -1) - 1
        gt_valid_len[gt_valid_len < 0] = 0
        gt_cum = gt.cumsum(-2)
        gt_end_point = torch.gather(gt_cum, dim=-2,
                                    index=gt_valid_len.reshape(*gt_valid_len.shape, 1, 1).repeat(1, 1, 1, 2))

        pred_cum = pred.cumsum(dim=-2)
        pred_endpoint = torch.gather(pred_cum, dim=-2,
                                     index=gt_valid_len.reshape(*gt_valid_len.shape, 1, 1, 1).repeat(1, 1,
                                                                                                     pred.shape[2], 1,
                                                                                                     2)).squeeze(-2)

        #dis_mat = torch.norm((gt_end_point - pred_endpoint), p=2, dim=-1)  # [batch, nbrs_num+1, K]
        dis_mat, in_mask, valid_class = self.calDist(pred_endpoint, gt_end_point, gt_valid_len, data['misc'])
        index = torch.argmin(dis_mat, dim=-1)  # [batch, nbrs_num+1, 1]

        tracks_to_predict = data['tracks_to_predict']
        valid_class = valid_class*tracks_to_predict
        cls_loss = self.maxEntropyLoss(tracks_to_predict, conf, dis_mat)  # Margin Loss
        #cls_loss = self.crossEntropyLoss(conf,in_mask,valid_class)
        reg_loss = self.huber_loss(tracks_to_predict, pred, gt, index, gt_mask)  # Huber Loss

        in_mask = in_mask.sum(dim=-1).to(bool)
        mr = 1-(in_mask*valid_class).sum()/valid_class.sum().float()
        #miss_rate = self.cal_total_miss_rate(tracks_to_predict, gt_end_point, pred_endpoint)

        losses = {'reg_loss': reg_loss, 'cls_loss': cls_loss}

        loss = self.K * reg_loss + cls_loss

        return loss, losses, mr

    def crossEntropyLoss(self,conf,in_mask,valid_class):

        loss = torch.nn.BCELoss(reduction='none')
        cls_loss = loss(conf, in_mask.to(torch.float32))
        cls_loss = cls_loss.mean(dim=-1)
        cls_loss = (cls_loss*valid_class).sum()/valid_class.sum()
        return cls_loss

    def calDist(self, pred_endpoint, gt_end_point, gt_valid_len, data):
        # get endpoint gt'yaw

        yaw = data[..., 10:, 4]
        vel = torch.square(data[..., 10, 5]**2+data[..., 10, 6]**2)
        start_yaw = yaw[..., 0]

        mask = gt_valid_len >= 49
        end_yaw = torch.gather(yaw, dim=-1, index=gt_valid_len.unsqueeze(-1)).squeeze(-1)
        pred_endpoint = self.rotate(pred_endpoint, -start_yaw)
        gt_end_point = self.rotate(gt_end_point, -start_yaw)

        thres = vel.unsqueeze(-1).repeat(1,1,2)
        thres[...,0] = 1.5+1.5*(thres[...,0]-1.4)/9.6
        thres[..., 1] = 3+3 * (thres[..., 1] - 1.4) / 9.6

        thres[...,0][thres[..., 0]<1.5] = 1.5
        thres[..., 0][thres[..., 0]>3] = 3
        thres[..., 1][thres[..., 1]<3] = 3
        thres[..., 1][thres[..., 1]>6] = 6

        # rotate prediction to gt end point's coord
        pred_endpoint-=gt_end_point
        pred_endpoint = abs(self.rotate(pred_endpoint,end_yaw))
        thres = thres.unsqueeze(-2).repeat(1,1,pred_endpoint.shape[-2],1)
        x_in = pred_endpoint[...,0]<thres[...,0]
        y_in = pred_endpoint[...,1]<thres[...,1]
        all_in = x_in*y_in
        dist = torch.square(4*pred_endpoint[...,0]**2+pred_endpoint[...,1]**2)
        return dist, all_in, mask

    def maxEntropyLoss(self, predict_flag, score, dis_mat):
        '''
            params:
                score[batch, num agents, K]: confidence of each traj
                dis_mat[batch, num agents, K]: the score beased on endpoint
                endpoint_exist_mask[batch, num agents,1]: contain the info of which traj has endpoint
        '''

        dis_mat = dis_mat.detach()
        target = torch.nn.Softmax(dim=-1)(-dis_mat / self.alpha)
        KL_loss = torch.nn.KLDivLoss(reduction='none')(score, target)

        predict_flag = predict_flag.unsqueeze(-1)
        cls_loss_sum = (KL_loss * predict_flag).sum()
        agent_sum = max(predict_flag.sum(), 1)
        cls_loss = cls_loss_sum / agent_sum
        return cls_loss

    def huber_loss(self, predict_flag, pred, gt, expected_traj_index, gt_mask):
        '''
            params:
                score: the predicted traj
                target: the ground truth of traj
                expected_traj_index: decide which traj can be choose from total K trajs.
                mask[batch, numagents, future num frame]: the frame avaliable mask
        '''
        expected_traj_index = expected_traj_index.view(*expected_traj_index.shape, 1, 1, 1).repeat(
            (1, 1, 1, *pred.shape[3:]))  # [batch_size, nbrs_num+1, K, 30, 2]
        best_pred = torch.gather(pred, dim=-3, index=expected_traj_index).squeeze(2)
        reg_loss = torch.nn.SmoothL1Loss(reduction='none')(best_pred, gt).mean(-1)

        loss_sum = ((reg_loss * gt_mask).sum(dim=-1)) * predict_flag
        gt_sum = gt_mask.sum(dim=-1)
        gt_sum[gt_sum == 0] = 1
        mean_agent_loss = loss_sum / gt_sum
        agent_sum = max(predict_flag.sum(), 1)
        reg_loss = mean_agent_loss.sum() / agent_sum
        return reg_loss

    def multi_task_gather(self, losses):
        # Total Loss
        if self.cfg['loss_params']['MultiTask']:
            total_loss = 0
            for i, loss in enumerate(losses.values()):
                loss = loss['loss']
                # ? which kinds of multi task loss suits our task.s
                total_loss += (2 * self.auxiliary_params[i]) ** -2 * loss + torch.log(self.auxiliary_params[i] ** 2 + 1)
        else:
            total_loss = self.loss_cfg['k'] * losses['reg_loss']['loss'] + losses['cls_loss']['loss']

        losses_text = ''
        for loss_name in losses:
            losses_text += loss_name + ':{:.3f}-'.format(losses[loss_name]['loss'])

        return total_loss, losses_text

    def cal_total_miss_rate(self, predict_flag, target_endpoint, prediction_endpoint):
        '''
            predict_flag:       [batch, car]
            prediction_endpoint: [batch, nbrs_num+1, K, 2]
            target_endpoint:    [batch, nbrs_num+1, 1, 2]
        '''

        inside_mask = (torch.norm(target_endpoint - prediction_endpoint, p=2, dim=-1) < 8.0)
        missed = (inside_mask.sum(dim=(-1)) == 0) * predict_flag
        mr = missed.sum() / max(predict_flag.sum().float(), 1)
        return mr

    def rotate(self, vec, yaw):
        c, s = torch.cos(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2]), torch.sin(yaw).unsqueeze(-1).repeat(1, 1, vec.shape[-2])
        vec[..., 0], vec[..., 1] = c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]
        return vec
