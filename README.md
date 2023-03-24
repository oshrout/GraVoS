# GraVoS: Voxel Selection for 3D Point-Cloud Detection
This repository contains the PyTorch implementation of the CVPR'2023 paper - [GraVoS: Voxel Selection for 3D Point-Cloud Detection](https://arxiv.org/abs/2208.08780).

## Installation
The code was tested in the following environment:
* Ubuntu 18.04/20.04
* Python 3.7
* CUDA 11.1
* Pytorch 1.9.0
* spconv 2.1.21
* spconv-cu111 2.1.21

## Create enviroment
```
conda create --name GraVoS python==3.7
conda activate GraVoS
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/oshrout/GraVoS
cd GraVoS
pip install -r requirements.txt
```

## Install spconv and OpenPCDet
1. Install spconv with pip, see [spconv](https://github.com/traveller59/spconv) for more details.
2. Install pcdet by running `python setup.py develop`, see [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for more details.

## License
This project is realeased under the [Apache License 2.0](https://github.com/oshrout/GraVoS/blob/master/LICENSE).

## Citation
If you find this project useful, please consider cite:
```
@article{shrout2022gravos,
  title={GraVoS: Gradient based Voxel Selection for 3D Detection},
  author={Shrout, Oren and Ben-Shabat, Yizhak and Tal, Ayellet},
  journal={arXiv preprint arXiv:2208.08780},
  year={2022}
}
```

## Acknowlegements
Our code is mostly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
