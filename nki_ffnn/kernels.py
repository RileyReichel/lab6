import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_
"""
CS152 Lab 6 - Directed Portion
NKI Kernels for Feedforward Neural Network

This file contains the NKI kernel implementations for:
  - nki_transpose: Transpose a 2D tensor
  - nki_bias_add_act: Add bias and apply activation (relu or softmax)
  - nki_forward: Forward pass of a 2-layer FFNN
  - nki_predict: Forward pass + argmax prediction
"""
@nki.jit
def nki_transpose(in_tensor):
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows
    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)
    TILE_SIZE = nl.tile_size.pmax
    for i_row in nl.affine_range(i_rows // TILE_SIZE):
        for i_col in nl.affine_range(i_cols // TILE_SIZE):
            tile = nl.load_transpose2d(in_tensor[i_row * TILE_SIZE : (i_row + 1) * TILE_SIZE, i_col * TILE_SIZE : (i_col + 1) * TILE_SIZE])
            nl.store(out_tensor[i_col * TILE_SIZE : (i_col + 1) * TILE_SIZE, i_row * TILE_SIZE : (i_row + 1) * TILE_SIZE], value=tile)
    return out_tensor
 
 
@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)
    TILE_SIZE = nl.tile_size.pmax
    for i_batch in nl.affine_range(BATCH_SIZE // TILE_SIZE):
        if act == 'relu':
            for i_hidden in nl.affine_range(HIDDEN_SIZE // TILE_SIZE):
                a_tile = nl.load(A[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE])
                b_tile = nl.load(b[0:1, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE])
                res = nl.add(a_tile, b_tile)
                res = nl.relu(res)
                nl.store(result[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE], value=res)
        elif act == 'softmax':
            a_row = nl.load(A[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, 0 : HIDDEN_SIZE])
            b_row = nl.load(b[0:1, 0 : HIDDEN_SIZE])
            res = nl.add(a_row, b_row)
            max_val = nl.max(res, axis=1, keepdims=True)
            res = nl.subtract(res, max_val)
            res = nl.exp(res)
            sum_val = nl.sum(res, axis=1, keepdims=True)
            res = nl.divide(res, sum_val)
            nl.store(result[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, 0 : HIDDEN_SIZE], value=res)
    return result
 
 
@nki.jit
def nki_forward(X, W1, b1, W2, b2, matmul_kernel='tiled'):
    if matmul_kernel == 'tiled':
        nki_matmul = nki_matmul_tiled_
    elif matmul_kernel == 'hoist_load':
        nki_matmul = nki_matmul_hoist_load_
    elif matmul_kernel == 'block_free_dimension':
        nki_matmul = nki_matmul_block_free_dimension_
    elif matmul_kernel == 'fully_optimized':
        nki_matmul = nki_matmul_fully_optimized_
    else:
        raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")
    XT = nki_transpose(X)
    H = nki_matmul(XT, W1)
    H = nki_bias_add_act(H, b1, act='relu')
    HT = nki_transpose(H)
    probs = nki_matmul(HT, W2)
    probs = nki_bias_add_act(probs, b2, act='softmax')
    return probs
 
 
@nki.jit
def nki_predict(X, W1, b1, W2, b2, matmul_kernel='tiled'):
    probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel=matmul_kernel)
    BATCH_SIZE, OUTPUT_SIZE = probs.shape
    predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)
    TILE_SIZE = nl.tile_size.pmax
    for i_batch in nl.affine_range(BATCH_SIZE // TILE_SIZE):
        probs_tile = nl.load(probs[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, 0 : OUTPUT_SIZE])
        max_vals = nisa.tensor_reduce(np.max, probs_tile, axis=(1,), keepdims=True)
        idx_tile = nl.ndarray((TILE_SIZE, OUTPUT_SIZE), dtype=np.float32, buffer=nl.sbuf)
        nisa.iota(idx_tile, pattern=[[1, OUTPUT_SIZE]], offset=0)
        scored = nl.subtract(nisa.tensor_scalar(probs_tile, nl.multiply, 1e6), idx_tile)
        max_scored = nisa.tensor_reduce(np.max, scored, axis=(1,), keepdims=True)
        score_diff = nl.subtract(scored, max_scored)
        abs_score_diff = nl.abs(score_diff)
        penalty = nisa.tensor_scalar(abs_score_diff, nl.multiply, 1e10)
        final_idx = nl.add(idx_tile, penalty)
        argmax_idx = nisa.tensor_reduce(np.min, final_idx, axis=(1,), dtype=np.int32)
        nl.store(predictions[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE], value=argmax_idx)
    return predictions
 
