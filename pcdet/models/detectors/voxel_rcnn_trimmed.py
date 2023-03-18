from .detector3d_template import Detector3DTemplate


class VoxelRCNNtrimmed(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if self.training:
            batch_dict['voxels'].requires_grad=True
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            _, tb_dict = self.get_training_loss()
            batch_dict['tb_dict'] = tb_dict
            return batch_dict
        else:
            print(f'Not implemented')
            raise NotImplementedError


    def get_training_loss(self):
        loss_rpn, tb_dict = self.dense_head.get_loss()
        return loss_rpn, tb_dict
