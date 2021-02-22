# Lambda Networks in PyTorch  ![Tests Passing](https://github.com/titu1994/lambda_networks_pt/actions/workflows/python-package.yml/badge.svg)

Lambda Networks from the paper [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://arxiv.org/abs/2102.08602) by Bello et al.
Adds support for Rank 3 and Rank 5 tensor inputs as well as initial implementation of ResNet-D with Lambda Convolutions.

Code is adapted from `lucidrains` implementation - https://github.com/lucidrains/lambda-networks

TODO:
  - Add support for ResNet-RS

# Installation

```
pip install --upgrade git+https://github.com/titu1994/lambda_networks_pt.git
```

# Usage

There are three modules inside -

- `lambda_module_1d.py`: Implements Lambda Network block for Rank 3 input data (B, C, T)
- `lambda_module_2d.py`: Implements Lambda Network block for Rank 4 input data (B, C, H, W)
- `lambda_module_3d.py`: Implements Lambda Network block for Rank 5 input data (B, C, D, H, W)
- `lambda_resnets.py`  : Implements Lambda ResNets (Using ResNet-D, Not ResNet-RS!) for Rank 4 input data (B, C, H, W)


```python
import lambda_networks

# 1D Block, 2D Block or 3D Block
module = lambda_networks.LambdaLayer1D(
    dim=32, 
    dim_out=64,
    m=None,  # Use positive integer for Global Context using Lambda Layer. Represents "m" in the paper.
    r=None,  # Use positive integer for Local Context using Lambda Convolution. Represents "r" in the paper.
    dim_k=16,  # Dimension of key/query.
    dim_intra=1  # Intra-dimension "u" in the paper.
    heads=4,  # Number of heads. Represents "h" in the paper.
    implementation=0,  # Defaults to 0 generally, which implements the paper version of n-D Lambda using (n+1)-D Convolution.
)

# Lambda ResNet-D
model = lambda_networks.resnet_18(
    lambda_m: bool = False,  # Bool flag whether global context should be used or not. If set to True, pass in `input_size` as well to compute global context size per block.
    lambda_r: Optional[int] = None,  # Optional int, which if passed computes Local Context using Lambda Convolution. 
    lambda_k: int = 16,  # Dimension of key/query.
    lambda_u: int = 1,  # Intra-dimension "u" in the paper.
    lambda_heads: int = 4,  #  Number of heads. Represents "h" in the paper.
    input_size: Optional[int] = None,  # Optional int representing Height and Width of the image. Must be passed if `lambda_m` is set to True.
)

```

# Tests

Tests will perform CPU only checks if there are no GPUs. If GPUs are present, will run all tests once for `cuda:0` as well.

```bash
pytest tests/
```

# Requirements

- pytorch >= 1.7.1. Older versions might work, not tested.
- einops - Required for LambdaLayer2D/3D and Lambda-ResNet-D. Not required for LambdaLayer1D.
