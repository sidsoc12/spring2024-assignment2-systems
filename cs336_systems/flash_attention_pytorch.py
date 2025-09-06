import torch
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from the input tensors
        # We start by checking the number of dimensions of the input tensor.
        input_is_3d = Q.ndim == 3

        # If the input is 3D, we add a dummy "NUM_HEADS" dimension of size 1.
        # This makes the rest of our code work without changes, as it can
        # consistently expect a 4D tensor.
        if input_is_3d:
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        # Now, we can safely unpack the 4D shape.
        BATCH_SIZE, NUM_HEADS, N_QUERIES, D = Q.shape
        _, _, N_KEYS, _ = K.shape
       

        # Determine tile sizes. As per the prompt, we'll use a fixed size.
        # In a real implementation, this would be tuned for the specific hardware.
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128

        # Calculate the number of tiles
        Tq = math.ceil(N_QUERIES / Q_TILE_SIZE)
        Tk = math.ceil(N_KEYS / K_TILE_SIZE)
        
        # The scale factor is 1/sqrt(d)
        scale = D ** -0.5
        
        # Initialize the output tensor O and the log-sum-exp tensor L for the backward pass
        O = torch.zeros_like(Q)
        L = torch.zeros(BATCH_SIZE, NUM_HEADS, N_QUERIES, device=Q.device)

        # --- The Tiled Computation ---
        # This is a direct implementation of Algorithm 1 from the paper.
        
        # Outer loop over query tiles
        for i in range(Tq):
            # 1. Load the i-th query tile from HBM into SRAM (conceptually)
            q_tile_start = i * Q_TILE_SIZE
            q_tile_end = min((i + 1) * Q_TILE_SIZE, N_QUERIES)
            Qi = Q[:, :, q_tile_start:q_tile_end, :]
            
            # 2. Initialize running accumulators in SRAM (conceptually)
            O_i = torch.zeros_like(Qi, dtype=torch.float32)
            l_i = torch.zeros(BATCH_SIZE, NUM_HEADS, q_tile_end - q_tile_start, device=Q.device, dtype=torch.float32)
            m_i = torch.full((BATCH_SIZE, NUM_HEADS, q_tile_end - q_tile_start), float('-inf'), device=Q.device, dtype=torch.float32)

            # Inner loop over key/value tiles
            for j in range(Tk):
                # 3. Load the j-th key and value tiles from HBM
                k_tile_start = j * K_TILE_SIZE
                k_tile_end = min((j + 1) * K_TILE_SIZE, N_KEYS)
                Kj = K[:, :, k_tile_start:k_tile_end, :]
                Vj = V[:, :, k_tile_start:k_tile_end, :]

                # 4. Compute the tile of attention scores, S_ij
                S_ij = torch.einsum('b h i d, b h j d -> b h i j', Qi, Kj) * scale
                
                # Causal masking (if enabled)
                if is_causal:
                    # Create a mask for the current tile
                    row_indices = torch.arange(q_tile_start, q_tile_end, device=Q.device).view(1, 1, -1, 1)
                    col_indices = torch.arange(k_tile_start, k_tile_end, device=Q.device).view(1, 1, 1, -1)
                    mask = row_indices >= col_indices
                    S_ij = torch.where(mask, S_ij, float('-inf'))

                # 5. Compute the new running maximum, m_new
                m_ij = torch.max(S_ij, dim=-1)[0]
                m_new = torch.maximum(m_i, m_ij)
                
                # 6. Compute the unnormalized probabilities, P_tilde
                P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
                
                # 7. Compute the new denominator, l_new
                # This is the "online softmax" update step
                alpha = torch.exp(m_i - m_new)
                l_new = alpha.unsqueeze(-1) * l_i.unsqueeze(-1) + torch.sum(P_tilde_ij, dim=-1, keepdim=True)
                
                # 8. Compute the new output tile, O_i
                # This is the "online output" update step
                O_i = alpha.unsqueeze(-1) * O_i
                O_i += torch.matmul(P_tilde.to(Vj_tile.dtype), Vj_tile)
                
                # Update the running accumulators for the next iteration
                l_i = l_new.squeeze(-1)
                m_i = m_new

            # 9. After iterating through all key tiles, normalize the output tile
            O_i_normalized = O_i / l_i.unsqueeze(-1)
            
            # 10. Write the final output tile and L_i back to HBM (conceptually)
            O[:, :, q_tile_start:q_tile_end, :] = O_i_normalized.to(Q.dtype)
            L[:, :, q_tile_start:q_tile_end] = m_i + torch.log(l_i)
        
        # Save necessary tensors for the backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

    @staticmethod
    def backward(ctx, grad_output):
        # For now, as per the assignment, we just raise an error.
        # The backward pass will be implemented later.
        raise NotImplementedError("FlashAttention backward pass is not implemented yet.")