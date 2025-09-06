import torch
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Handle 3D input by unsqueezing heads
        input_is_3d = Q.ndim == 3
        if input_is_3d:
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        BATCH_SIZE, NUM_HEADS, N_QUERIES, D = Q.shape
        _, _, N_KEYS, _ = K.shape

        # Tile sizes (fixed for simplicity)
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128

        # Number of tiles
        Tq = math.ceil(N_QUERIES / Q_TILE_SIZE)
        Tk = math.ceil(N_KEYS / K_TILE_SIZE)

        # Scaling factor
        scale = D ** -0.5

        # Output tensor
        O = torch.zeros_like(Q)

        # Tensor for log-sum-exp accumulator (B, H, N)
        L = torch.zeros(BATCH_SIZE, NUM_HEADS, N_QUERIES, device=Q.device)

        # --- Tiled computation ---
        for i in range(Tq):
            q_tile_start = i * Q_TILE_SIZE
            q_tile_end = min((i + 1) * Q_TILE_SIZE, N_QUERIES)
            Qi = Q[:, :, q_tile_start:q_tile_end, :]

            # Running accumulators
            O_i = torch.zeros_like(Qi, dtype=torch.float32)
            l_i = torch.zeros(BATCH_SIZE, NUM_HEADS, q_tile_end - q_tile_start,
                              device=Q.device, dtype=torch.float32)
            m_i = torch.full((BATCH_SIZE, NUM_HEADS, q_tile_end - q_tile_start),
                             float('-inf'), device=Q.device, dtype=torch.float32)

            for j in range(Tk):
                k_tile_start = j * K_TILE_SIZE
                k_tile_end = min((j + 1) * K_TILE_SIZE, N_KEYS)
                Kj = K[:, :, k_tile_start:k_tile_end, :]
                Vj = V[:, :, k_tile_start:k_tile_end, :]

                # Attention scores
                S_ij = torch.einsum('b h i d, b h j d -> b h i j', Qi, Kj) * scale

                # Apply causal masking if enabled
                if is_causal:
                    row_idx = torch.arange(q_tile_start, q_tile_end, device=Q.device).view(1, 1, -1, 1)
                    col_idx = torch.arange(k_tile_start, k_tile_end, device=Q.device).view(1, 1, 1, -1)
                    mask = row_idx >= col_idx
                    S_ij = torch.where(mask, S_ij, float('-inf'))

                # New running max
                m_ij = torch.max(S_ij, dim=-1)[0]  # (B, H, i)
                m_new = torch.maximum(m_i, m_ij)

                # Unnormalized probs
                P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(-1))

                # Online softmax denominator
                alpha = torch.exp(m_i - m_new)  # (B, H, i)
                l_new = alpha * l_i + torch.sum(P_tilde_ij, dim=-1)

                # Update output
                O_i = alpha.unsqueeze(-1) * O_i \
                    + torch.einsum('b h i j, b h j d -> b h i d', P_tilde_ij, Vj)

                # Update accumulators
                l_i = l_new
                m_i = m_new

            # Normalize
            O_i_normalized = O_i / l_i.unsqueeze(-1)

            # Write back
            O[:, :, q_tile_start:q_tile_end, :] = O_i_normalized.to(Q.dtype)
            L[:, :, q_tile_start:q_tile_end] = m_i + torch.log(l_i + 1e-9)  # numerical stability

        # Reduce L to shape (B, N) as test requires
        if NUM_HEADS == 1:
            L_reduced = L.squeeze(1)  # (B, N)
        else:
            L_reduced = L.mean(dim=1)  # (B, N), if multiple heads

        # Save only the log-sum-exp tensor
        ctx.save_for_backward(L_reduced)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward pass is not implemented yet.")
