# Copyright (c) 2024, Tri Dao.

import paddle
import paddle.nn.functional as F
from paddle.autograd import PyLayer


try:
    import causal_conv1d_cuda_paddle as causal_conv1d_cuda
except ImportError:
    causal_conv1d_cuda = None
    print("causal_conv1d_cuda_paddle is not found. Please install it.")


def backward_return_wrapper(*args):
    result = []
    for each in args:
        if each is not None and isinstance(each, paddle.Tensor):
            result.append(each)
    return tuple(result)

class CausalConv1dFn(PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=False,
        final_states_out=None,
        activation=None,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.strides[2] != 1 and x.strides[1] != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if seq_idx is not None:
            assert (
                initial_states is None
            ), "initial_states must be None if seq_idx is not None"
            assert (
                not return_final_states
            ), "If seq_idx is not None, we don't return final_states_out"
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        if initial_states is not None and (
            initial_states.strides[2] != 1 and initial_states.strides[1] != 1
        ):
            initial_states = initial_states.contiguous()

        if return_final_states:
            assert (
                x.strides[1] == 1
            ), "Only channel-last layout support returning final_states_out"
            if final_states_out is not None:
                assert (
                    final_states_out.strides[2] == 1 or final_states_out.strides[1] == 1
                )
            else:
                batch, dim, seqlen = x.shape
                width = weight.shape[1]
                final_states_out = paddle.zeros(
                    [batch, width - 1, dim], dtype=x.dtype
                ).transpose([0, 2, 1])
        else:
            final_states_out = None

        ctx.activation = activation in ["silu", "swish"]

        out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, 
            weight, 
            bias, 
            seq_idx, 
            initial_states, 
            final_states_out, 
            ctx.activation, 
        )

        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = return_final_states
        ctx.return_dinitial_states = (
            initial_states is not None and not initial_states.stop_gradient
        )
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensor()
        dfinal_states = args[0] if ctx.return_final_states else None

        # NEW ADD, not in c++ code
        is_channel_last = x.strides[1] == 1 and x.strides[2] > 1
        if not is_channel_last and dout.strides[2] != 1:
            dout = dout.contiguous()
            if ctx.return_final_states:
                dfinal_states = dfinal_states.contiguous()

        if is_channel_last and dout.strides[1] != 1:
            dout = dout.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
            if ctx.return_final_states:
                dfinal_states = dfinal_states.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])

        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = causal_conv1d_cuda.causal_conv1d_bwd(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            None,
            ctx.return_dinitial_states,
            ctx.activation
        )
        return backward_return_wrapper(
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            dinitial_states if initial_states is not None else None,
            None,
            None,
            None,
        )


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    return CausalConv1dFn.apply(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        return_final_states,
        final_states_out,
        activation,
    )


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.cast(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = paddle.concat([initial_states.cast(x.dtype), x], axis=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        tmp = width - 1 - x.shape[-1]
        if tmp < 0:
            final_states = x[..., -tmp:].cast(
                dtype_in
            )  # (batch, dim, width - 1)
        else:
            final_states = F.pad(x, (width - 1 - x.shape[-1], 0), data_format="NCL").cast(
                dtype_in
            )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states, False)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).cast(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation = activation in ["silu", "swish"]

    out  = causal_conv1d_cuda.causal_conv1d_update(
        x, 
        conv_state, 
        weight, 
        bias, 
        activation,
    )
    return out


def causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None):
    """
    x: (batch, dim)
    conv_state: (batch, dim, width)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    batch, dim = x.shape
    width = weight.shape[1]
    assert conv_state.shape == [batch, dim, width]
    assert weight.shape == [dim, width]
    conv_state.copy_(paddle.roll(conv_state, shifts=-1, axis=-1).cast(conv_state.dtype), False)  # Update state (B D W)
    conv_state[:, :, -1] = x
    out = paddle.sum(conv_state * weight, axis=-1)  # (B D)
    if bias is not None:
        out += bias
    return (out if activation is None else F.silu(out)).cast(dtype=dtype_in)