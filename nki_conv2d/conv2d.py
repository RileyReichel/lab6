import os
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

"""
Performs a 2D convolution operation using NKI.
Args:
    X: Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W: Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias: Bias tensor of shape (out_channels).
Returns:
    out_tensor: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    For ease of implementation, you can expect the inputs to abide by the following restrictions
    - filter_height == filter_width
    - input_channels % 128 == 0
    - output_channels % 128 == 0
    - output_width * output_height % 512 == 0
"""
@nki.jit
def conv2d_nki(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
 
    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1
 
    assert filter_height == filter_width, "Filter height must be equal to filter width"
    assert in_channels % 128 == 0, "Input channels must be divisible by 128"
    assert out_channels % 128 == 0, "Output channels must be divisible by 128"
    assert out_width * out_height % 512 == 0, "Output width * output height must be divisible by 512"
 
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
 
    c_in_tile = nl.tile_size.pmax
    c_out_tile = nl.tile_size.gemm_stationary_fmax
    n_tiles_c_in = in_channels // c_in_tile
    n_tiles_c_out = out_channels // c_out_tile
 
    for img in nl.affine_range(batch_size):
        for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
            bias_tile = nl.load(bias[c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile])
            bias_tile_2d = bias_tile.reshape((c_out_tile, 1))
 
            for out_row in nl.affine_range(out_height):
                row_out = nl.zeros((c_out_tile, out_width), dtype=nl.float32, buffer=nl.psum)
 
                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    w_all = nl.ndarray((c_out_tile, c_in_tile, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w_all, src=W[c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile, c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile, 0:filter_height, 0:filter_width])
 
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            w_tile_t = nisa.nc_transpose(w_all[:, :, i, j])
                            x_tile = nl.load(X[img, c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile, i + out_row, j : j + out_width])
                            row_out += nisa.nc_matmul(w_tile_t, x_tile)
 
                row_out_sbuf = nl.copy(row_out, dtype=X.dtype)
                row_out_sbuf = nl.add(row_out_sbuf, bias_tile_2d)
                nl.store(X_out[img, c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile, out_row, 0 : out_width], value=row_out_sbuf)
 
    return X_out
 
