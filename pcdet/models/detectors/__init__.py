from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint

from .voxel_rcnn_trimmed import VoxelRCNNtrimmed
from .second_net_trimmed import SECONDNettrimmed
from .centerpoint_trimmed import CenterPointtrimmed
from .PartA2_net_trimmed import PartA2Nettrimmed


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'SECONDNettrimmed': SECONDNettrimmed,
    'PartA2Net': PartA2Net,
    'PartA2Nettrimmed': PartA2Nettrimmed,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'VoxelRCNNtrimmed': VoxelRCNNtrimmed,
    'CenterPoint': CenterPoint,
    'CenterPointtrimmed': CenterPointtrimmed
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
