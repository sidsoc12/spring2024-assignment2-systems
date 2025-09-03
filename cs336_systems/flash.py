import torch
import triton 
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block = tl.advance(K_block_ptr, (0, lo))
    V_block = tl.advance(V_block_ptr, (0, lo))

    # loop over k,v blocks in parallel 
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # compute qk 
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block) # K already transposed

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i
    


@triton.jit
def _attn_fwd(
    Q, # this is a pointer to the Q matrix 
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,

    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE:tl.constexpr
):

    # figure out which block of Q and which head and which batch we are processing 
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS 
    index_head  = index_batch_head % NUM_HEADS 

    # since Q,K,V are pointers to the start of the matrix-array (as stored in memory), we need to get the actual values from the matrix (basically getting the query matrix of Q for the specific batch and head of this program)

    qvk_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head 

    # make a pointer to the Q block that we are processing 
    Q_block_ptr = tl.make_block_ptr(
        base = Q + qvk_offset, # query matrix for the specific batch and head of this program 
        shape = [SEQ_LEN, HEAD_DIM], # shape of the query matrix 
        strides = [stride_Q_seq, stride_Q_dim], # strides of the query matrix 
        offsets = [block_index_q * BLOCK_SIZE_Q, 0],  # defines the starting corner of the block
        block_shape = [BLOCK_SIZE_Q, HEAD_DIM], # shape of the block 
        order = [1,0], # order of the block 
    )

    # get the pointer to the entire V matrix for the specific batch and head (since each head in each does this parallely)
    V_block_ptr = tl.make_block_ptr(
        base = V + qvk_offset, # value matrix for the specific batch and head of this program 
        shape = [SEQ_LEN, HEAD_DIM], # shape of the value matrix 
        strides = [stride_V_seq, stride_V_dim], # strides of the value matrix 
        offsets = [0, 0],  #defines the starting corner of the very first block 
        block_shape = [BLOCK_SIZE_KV, HEAD_DIM], # shape of the block 
        order = [1,0], # order of the block 
    )
    
    # pointer to transposed K matrix for specific batch and head 
    K_block_ptr = tl.make_block_ptr(
        base = K + qvk_offset,
        shape = [HEAD_DIM, SEQ_LEN],
        strides = (stride_K_dim, stride_K_seq),
        offsets = [0,0],
        block_shape = [HEAD_DIM, BLOCK_SIZE_KV],
        order = [1,0],
    )

    O_block_ptr = tl.make_block_ptr(
        base = O + qvk_offset,
        shape = [SEQ_LEN, HEAD_DIM],
        strides = [stride_O_seq, stride_O_dim],
        offsets = [block_index_q * BLOCK_SIZE_Q, 0],
        block_shape = [BLOCK_SIZE_Q, HEAD_DIM],
        order = [1,0],
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) 
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Initializations for each computation using each query block 

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) - float('-inf') # initialize maximum of each row to -inf
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) + 1.0 # running sum of each row to 1.0

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype = tl.float32) # initialize output block to 0 

    # load blocks of Q to SRAM 
    Q_block = tl.load(Q_block_ptr)

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

         # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


  
    

    









class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V  = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V

        # define output matrix
        O = torch.empty_like(Q)
        stage = 3 if casual else 1 

        # define program grid (how many independent parallel programs do we want to run)

        grid = lambda args: (
            # how many blocks of Q do we have 
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), 
            # how many heads and batches we have 
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # number of parallel programs = # batch x # head X (# tokens / # Qblock_size)

        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype = torch.float32)

        # call kernel for grid # of parallel programs (triton handles thread management internally within each block)
        # basically defining the amount of parallel programs/units we want to run 
        _attn_fwd[grid](
            Q = Q,
            K = K,
            V = V,
            softmax_scale= softmax_scale,
            M=M,
            O=O,
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stride(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stride(3),
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_dim = O.stride(3),
            BATCH_SIZE = Q.shape[0],
            NUM_HEADS = Q.shape[1],
            SEQ_LEN = Q.shape[2],
            HEAD_DIM = HEAD_DIM_K,
            STAGE=stage, # causal or not causal attention 
        )


        # stuff that needs to be saved for backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid 
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causual = causal 
        return O 