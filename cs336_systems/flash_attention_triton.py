import torch
import triton
import triton.language as tl
import math
from typing import Type

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    stride_lb, stride_lh, stride_lq,
    N_QUERIES, N_KEYS, NUM_HEADS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr, # Added is_causal flag
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Causal masking
    # Create the query index vector for this tile
    q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    

    for head_index in range(NUM_HEADS):
        q_base_ptr = Q_ptr + batch_index * stride_qb + head_index * stride_qh
        k_base_ptr = K_ptr + batch_index * stride_kb + head_index * stride_kh
        v_base_ptr = V_ptr + batch_index * stride_vb + head_index * stride_vh
        o_base_ptr = O_ptr + batch_index * stride_ob + head_index * stride_oh
        l_base_ptr = L_ptr + batch_index * stride_lb + head_index * stride_lh

        Q_block_ptr = tl.make_block_ptr(
            base=q_base_ptr, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        K_block_ptr_base = tl.make_block_ptr(
            base=k_base_ptr, shape=(D, N_KEYS), strides=(stride_kd, stride_kk),
            offsets=(0, 0), block_shape=(D, K_TILE_SIZE), order=(0, 1),
        )
        V_block_ptr_base = tl.make_block_ptr(
            base=v_base_ptr, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(0, 0), block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        
        O_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
        l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
        m_i = tl.full([Q_TILE_SIZE], value=float('-inf'), dtype=tl.float32)
        q_tile = tl.load(Q_block_ptr)

        num_kv_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        for j in range(num_kv_tiles):
            k_tile_start = j * K_TILE_SIZE
            K_block_ptr = tl.advance(K_block_ptr_base, (0, k_tile_start))
            V_block_ptr = tl.advance(V_block_ptr_base, (k_tile_start, 0))
            k_tile = tl.load(K_block_ptr)
            S_ij = tl.dot(q_tile, k_tile) * scale

            # Casual masking-
            if is_causal:
                # Create the key index vector for this tile
                k_indices = k_tile_start + tl.arange(0, K_TILE_SIZE)
                # Create the 2D mask by comparing the vectors
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                # Apply the mask
                S_ij = tl.where(causal_mask, S_ij, -1e6)
            # ---------------------------------------------
            # Online softmax
            m_ij = tl.max(S_ij, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            P_tilde_ij = tl.exp(S_ij - m_new[:, None])
            l_new = alpha * l_i + tl.sum(P_tilde_ij, 1)
            v_tile = tl.load(V_block_ptr)
            P_tilde_ij = P_tilde_ij.to(v_tile.dtype)
            O_i = O_i * alpha[:, None]
            O_i = tl.dot(P_tilde_ij, v_tile, O_i)
            l_i, m_i = l_new, m_new

        # Normalize
        O_i = O_i / l_i[:, None]
        O_block_ptr = tl.make_block_ptr(
            base=o_base_ptr, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0), block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        tl.store(O_block_ptr, O_i.to(O_ptr.type.element_ty))
        # Log-sum-exp
        l_final = m_i + tl.log(l_i)
        L_ptr_final = l_base_ptr + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        tl.store(L_ptr_final, l_final)

# Backward pass 
def _flash_bwd(Q, K, V, O, L, dO, is_causal):
    # Get dimensions
    BATCH_SIZE, NUM_HEADS, N_QUERIES, D = Q.shape
    scale = D ** -0.5
    
    # Pre-compute D = rowsum(O * dO)
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    
    # Recompute S = QK^T / sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.ones(N_QUERIES, N_QUERIES, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(causal_mask, float('-inf'))
        
    # Recompute P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))
    
    # Compute dV = P^T @ dO
    dV = torch.matmul(P.transpose(-2, -1), dO)
    
    # Compute dP = dO @ V^T
    dP = torch.matmul(dO, V.transpose(-2, -1))
    
    # Compute dS = P * (dP - D)
    dS = P * (dP - D)
    
    # Compute dQ = dS @ K / sqrt(d)
    dQ = torch.matmul(dS, K) * scale
    
    # Compute dK = dS^T @ Q / sqrt(d)
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    
    return dQ, dK, dV

# We JIT-compile our PyTorch backward pass for performance
compiled_flash_bwd = torch.compile(_flash_bwd)

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False): # Flag has default value
        BATCH_SIZE, NUM_HEADS, N_QUERIES, D = Q.shape
        _, _, N_KEYS, _ = K.shape
        Q_TILE_SIZE, K_TILE_SIZE = 128, 128
        O = torch.empty_like(Q)
        L = torch.zeros(BATCH_SIZE, NUM_HEADS, N_QUERIES, device=Q.device, dtype=torch.float32)
        scale = D ** -0.5

        # Calculate the grid size
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), BATCH_SIZE)

        # Launch the kernel
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            N_QUERIES, N_KEYS, NUM_HEADS,
            scale,
            D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal, # Pass the flag to the kernel
        )

        # Save for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal # Save the flag for backward
        return O

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # Call compiled backward
        dQ, dK, dV = compiled_flash_bwd(Q, K, V, O, L, grad_output, is_causal)

        # Return grads for (Q, K, V, is_causal)
        return dQ, dK, dV, None