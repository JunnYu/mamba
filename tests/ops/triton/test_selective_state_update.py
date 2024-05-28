# Copyright (C) 2023, Tri Dao.


import paddle
import pytest


from mamba_ssm_paddle.ops.triton.selective_state_update import selective_state_update, selective_state_update_ref


@pytest.mark.parametrize("itype", [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float16])
@pytest.mark.parametrize("has_z", [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update(dim, dstate, has_z, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == paddle.float32 else (5e-3, 1e-2)
    if itype == paddle.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    paddle.seed(0)
    batch_size = 2
    state = paddle.randn([batch_size, dim, dstate], dtype=itype)
    x = paddle.randn([batch_size, dim], dtype=itype)
    dt = paddle.randn([batch_size, dim], dtype=itype)
    dt_bias = paddle.rand([dim,]) - 4.0
    A = -paddle.rand([dim, dstate]) - 1.0
    B = paddle.randn([batch_size, dstate])
    C = paddle.randn([batch_size, dstate])
    D = paddle.randn([dim,])
    if has_z:
        z = paddle.randn(x.shape, dtype=x.dtype)
    else:
        z = None
    state_ref = state.detach().clone()
    out = selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert paddle.allclose(state.cast("float32"), state_ref.cast("float32"), rtol=rtol, atol=atol)
    assert paddle.allclose(out.cast("float32"), out_ref.cast("float32"), rtol=rtol, atol=atol)
