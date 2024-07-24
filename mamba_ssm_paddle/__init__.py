__version__ = "2.2.2"

import paddle
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)
if not hasattr(paddle, "masked_fill"):
    paddle.masked_fill = masked_fill
if not hasattr(paddle.Tensor, "masked_fill"):
    paddle.Tensor.masked_fill = masked_fill

from mamba_ssm_paddle.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn