import torch
import numpy as np

def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True
    )[0]
    return points_grad


class VoxGrad(object):
    def __init__(self, batch1, batch2, sample_loss):
        super(VoxGrad).__init__()
        self.batch = [batch1, batch2]
        self.sample_loss1 = batch1['tb_dict'].get(sample_loss)
        self.sample_loss2 = batch2['tb_dict'].get(sample_loss)
        self.batch_size = batch1['batch_size']

    def return_voxels_grads(self):
        sample_loss_list = [self.sample_loss1, self.sample_loss2]
        model_idx_list = [idx for idx in range(len(sample_loss_list))]
        voxels_grad_list = []
        for sampling_loss, model_idx in zip(sample_loss_list, model_idx_list):
            grad_per_voxel = self.get_voxels_gradients(sampling_loss, model_idx)
            voxels_grad_list.append(grad_per_voxel)
        return voxels_grad_list

    @torch.no_grad()
    def get_voxels_gradients(self, sampling_loss, model_idx):
        batch_voxels = self.batch[model_idx]['voxels']
        batch_voxel_coords = self.batch[model_idx]['voxel_coords']
        grads = gradient(inputs=batch_voxels, outputs=sampling_loss)

        voxels_grad_list = []
        batch_indices = batch_voxel_coords[:, 0].long()
        for bs_idx in range(self.batch_size):
            bs_mask = (batch_indices == bs_idx)
            cur_grads = grads[bs_mask]  # (num_voxels, points_in_voxel, C)

            # take the mean of l2 norm of points
            grad_per_voxel = cur_grads.norm(dim=-1).mean(dim=-1)

            voxels_grad_list.append(grad_per_voxel)
        return voxels_grad_list


def sample_voxels_2models(batch_late, batch_early, cfg_sampler):
    batch_voxels = batch_late['voxels']
    batch_voxel_coords = batch_late['voxel_coords']
    batch_voxel_num_points = batch_late['voxel_num_points']
    batch_size = batch_late['batch_size']
    sample_loss = cfg_sampler.get('sample_loss')
    vs_ratio = cfg_sampler.get('voxel_sample_ratio')

    late_det_cfg = cfg_sampler['sample_method']['late_detector']
    early_det_cfg = cfg_sampler['sample_method']['early_detector']

    # get batch gradient magnitudes
    voxels_grad = VoxGrad(batch_late, batch_early, sample_loss)
    voxels_grad_list_late, voxels_grad_list_early = voxels_grad.return_voxels_grads()

    voxels_list = []
    voxel_coords_list = []
    voxel_num_points_list = []

    batch_indices = batch_voxel_coords[:, 0].long()
    for bs_idx in range(batch_size):
        bs_mask = (batch_indices == bs_idx)
        cur_voxels = batch_voxels[bs_mask]  # (num_voxels, points_in_voxel, C)
        cur_voxel_coords = batch_voxel_coords[bs_mask]
        cur_voxel_num_points = batch_voxel_num_points[bs_mask]
        voxels_grad_late = voxels_grad_list_late[bs_idx]
        voxels_grad_early = voxels_grad_list_early[bs_idx]

        # sample the voxels based on the method
        if late_det_cfg.get('method') == 'topk':
            inder = convert_str2float(late_det_cfg.get('intra_det_ratio'))
            hp_indices_l = torch.topk(voxels_grad_late, int(vs_ratio * inder * voxels_grad_late.shape[0])).indices
            lp_indices_l = torch.topk(voxels_grad_late, int((1 - vs_ratio * inder) * voxels_grad_late.shape[0]), largest=False).indices
        elif late_det_cfg.get('method') == 'mean':
            hp_indices_l = torch.where(voxels_grad_late >= voxels_grad_late.mean())[0]
            lp_indices_l = torch.where(voxels_grad_late < voxels_grad_late.mean())[0]
        elif late_det_cfg.get('method') == 'median':
            hp_indices_l = torch.where(voxels_grad_late >= torch.median(voxels_grad_late))[0]
            lp_indices_l = torch.where(voxels_grad_late < torch.median(voxels_grad_late))[0]
        else:
            raise NotImplementedError

        if early_det_cfg.get('method') == 'topk':
            inder = convert_str2float(early_det_cfg.get('intra_det_ratio'))
            hp_indices_e = torch.topk(voxels_grad_early, int(vs_ratio * inder * voxels_grad_early.shape[0])).indices
            lp_indices_e = torch.topk(voxels_grad_early, int((1 - vs_ratio * inder) * voxels_grad_early.shape[0]), largest=False).indices
        elif early_det_cfg.get('method') == 'mean':
            hp_indices_e = torch.where(voxels_grad_early >= voxels_grad_early.mean())[0]
            lp_indices_e = torch.where(voxels_grad_early < voxels_grad_early.mean())[0]
        elif early_det_cfg.get('method') == 'median':
            hp_indices_e = torch.where(voxels_grad_early >= torch.median(voxels_grad_early))[0]
            lp_indices_e = torch.where(voxels_grad_early < torch.median(voxels_grad_early))[0]
        else:
            raise NotImplementedError

        # take the union of the high-priority indices and the intersection of the low-priority
        hp_indices = np.array(hp_indices_l.tolist() + list(set(hp_indices_e.tolist()) - set(hp_indices_l.tolist())))
        lp_indices = np.array(list(set.intersection(set(lp_indices_l.tolist()), set(lp_indices_e.tolist()))))

        # take all the high-priority and fill the rest (vs_ratio * total - high-priority) with low-priority voxels
        num_of_voxels = cur_voxels.shape[0]
        if hp_indices.shape[0] <= (vs_ratio * num_of_voxels):
            chosen_voxels = cur_voxels[hp_indices]
            chosen_voxel_coords = cur_voxel_coords[hp_indices]
            chosen_voxel_num_points = cur_voxel_num_points[hp_indices]
            # add the rest from the low-priority voxels
            dif_num_of_voxels = int(vs_ratio * num_of_voxels) - hp_indices.shape[0]
            chosen_indices = np.random.choice(a=lp_indices, size=dif_num_of_voxels, replace=False)
            chosen_voxels = torch.vstack((chosen_voxels, cur_voxels[chosen_indices]))
            chosen_voxel_coords = torch.vstack((chosen_voxel_coords, cur_voxel_coords[chosen_indices]))
            chosen_voxel_num_points = torch.cat((chosen_voxel_num_points, cur_voxel_num_points[chosen_indices]), dim=0)
        else: # take all the high-priority voxels (up to vs_ratio * total)
            chosen_indices = np.random.choice(a=hp_indices, size=int(vs_ratio * num_of_voxels), replace=False)
            chosen_voxels = cur_voxels[chosen_indices]
            chosen_voxel_coords = cur_voxel_coords[chosen_indices]
            chosen_voxel_num_points = cur_voxel_num_points[chosen_indices]

        voxels_list.append(chosen_voxels)
        voxel_coords_list.append(chosen_voxel_coords)
        voxel_num_points_list.append(chosen_voxel_num_points)

    voxels = torch.cat(voxels_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    voxel_num_points = torch.cat(voxel_num_points_list, dim=0)
    return voxels, voxel_coords, voxel_num_points


def sample_voxels_1model(batch, cfg_sampler):
    batch_voxels = batch['voxels']
    batch_voxel_coords = batch['voxel_coords']
    batch_voxel_num_points = batch['voxel_num_points']
    sample_loss = cfg_sampler.get('sample_loss')
    vs_ratio = cfg_sampler.get('voxel_sample_ratio')

    late_det_cfg = cfg_sampler['sample_method']['late_detector']

    # compute the grads w.r.t. the input voxels
    grads = gradient(inputs=batch_voxels, outputs=batch['tb_dict'].get(sample_loss))

    voxels_list = []
    voxel_coords_list = []
    voxel_num_points_list = []
    batch_indices = batch_voxel_coords[:, 0].long()
    for bs_idx in range(batch['batch_size']):
        bs_mask = (batch_indices == bs_idx)
        cur_voxels = batch_voxels[bs_mask]  # (num_voxels, points_in_voxel, C)
        cur_voxel_coords = batch_voxel_coords[bs_mask]
        cur_voxel_num_points = batch_voxel_num_points[bs_mask]
        cur_grads = grads[bs_mask]

        # take the mean of l2 norm of points
        norm_grad = cur_grads.norm(dim=-1).mean(dim=-1)

        # sample the voxels based on the method
        if late_det_cfg.get('method') == 'topk':
            inder = convert_str2float(late_det_cfg.get('intra_det_ratio'))
            hp_indices = torch.topk(norm_grad, int(vs_ratio * inder * norm_grad.shape[0])).indices
            lp_indices = torch.topk(norm_grad, int((1 - vs_ratio * inder) * norm_grad.shape[0]), largest=False).indices
        elif late_det_cfg.get('method') == 'mean':
            hp_indices = torch.where(norm_grad >= norm_grad.mean())[0]
            lp_indices = torch.where(norm_grad < norm_grad.mean())[0]
        elif late_det_cfg.get('method') == 'median':
            hp_indices = torch.where(norm_grad >= torch.median(norm_grad))[0]
            lp_indices = torch.where(norm_grad < torch.median(norm_grad))[0]
        else:
            raise NotImplementedError

        # take all the high-priority and fill the rest (vs_ratio * total - high-priority) with low-priority voxels
        num_of_voxels = cur_voxels.shape[0]
        if hp_indices.shape[0] <= (vs_ratio * num_of_voxels):
            chosen_voxels = cur_voxels[hp_indices]
            chosen_voxel_coords = cur_voxel_coords[hp_indices]
            chosen_voxel_num_points = cur_voxel_num_points[hp_indices]

            # add the rest from the low-priority voxels
            dif_num_of_voxels = int(vs_ratio * num_of_voxels) - hp_indices.shape[0]
            random_indices = torch.randperm(lp_indices.shape[0])[:dif_num_of_voxels]
            chosen_indices = lp_indices[random_indices]
            chosen_voxels = torch.vstack((chosen_voxels, cur_voxels[chosen_indices]))
            chosen_voxel_coords = torch.vstack((chosen_voxel_coords, cur_voxel_coords[chosen_indices]))
            chosen_voxel_num_points = torch.cat((chosen_voxel_num_points, cur_voxel_num_points[chosen_indices]), dim=0)
        else:  # take all the high-priority voxels (up to vs_ratio * total)
            random_indices = torch.randperm(hp_indices.shape[0])[:int(vs_ratio * num_of_voxels)]
            chosen_indices = hp_indices[random_indices]
            chosen_voxels = cur_voxels[chosen_indices]
            chosen_voxel_coords = cur_voxel_coords[chosen_indices]
            chosen_voxel_num_points = cur_voxel_num_points[chosen_indices]

        voxels_list.append(chosen_voxels)
        voxel_coords_list.append(chosen_voxel_coords)
        voxel_num_points_list.append(chosen_voxel_num_points)

    voxels = torch.cat(voxels_list, dim=0)
    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    voxel_num_points = torch.cat(voxel_num_points_list, dim=0)
    return voxels, voxel_coords, voxel_num_points


def convert_str2float(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)
