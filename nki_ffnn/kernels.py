import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
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
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    TILE_SIZE = nl.tile_size.pmax

    for i_batch in nl.affine_range(BATCH_SIZE // TILE_SIZE):
        for i_hidden in nl.affine_range(HIDDEN_SIZE // TILE_SIZE):
            a_tile = nl.load(A[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE])
            b_tile = nl.load(b[0, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE])

            res = nl.add(a_tile, b_tile)

            if act == 'relu':
                res = nl.relu(res)
            elif act == 'softmax':
                max_val = nl.max(res, axis=1)
                res = nl.subtract(res, max_val)
                res = nl.exp(res)
                sum_val = nl.sum(res, axis=1)
                res = nl.divide(res, sum_val)

            nl.store(result[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, i_hidden * TILE_SIZE : (i_hidden + 1) * TILE_SIZE], value=res)

    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
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

  # Layer 1
  # YOUR CODE HERE  
  W1T = nki_transpose(W1)
  H = nki_matmul(W1T, X)
  H = nki_bias_add_act(H, b1, act='relu')

  # Layer 2 (output)
  # YOUR CODE HERE
  W2T = nki_transpose(W2)
  probs = nki_matmul(W2T, H)
  probs = nki_bias_add_act(probs, b2, act='softmax')

  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """

    # YOUR CODE HERE
  probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel=matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  TILE_SIZE = nl.tile_size.pmax

  for i_batch in nl.affine_range(BATCH_SIZE // TILE_SIZE):
        probs_tile = nl.load(probs[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE, 0 : OUTPUT_SIZE])

        max_vals = nisa.tensor_reduce(np.max, probs_tile, axis=1, keepdims=True)

        argmax_result = nisa.argmax(probs_tile, axis=1)

        argmax_indices = argmax_result[:, 0]

        nl.store(predictions[i_batch * TILE_SIZE : (i_batch + 1) * TILE_SIZE], value=argmax_indices)
  return predictions
