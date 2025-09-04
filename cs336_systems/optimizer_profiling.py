import argparse
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
import cs336_basics.model as model_module
import math

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW # <-- IMPORT ADAMW

# --- Model Config for 'small' model ---
MODEL_CONFIG = {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12}

# --- Annotated Attention Function (from before) ---
@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        output = torch.einsum("...qk,...kv->...qv", attention_weights, V)
    return output

def profile_optimizer_step():
    """Profiles a single training step including the optimizer."""
    # --- Setup ---
    device = "cuda:0"
    torch.set_default_device(device)
    model = BasicsTransformerLM(vocab_size=10000, context_length=512, rope_theta=10000, **MODEL_CONFIG)
    model.to(device)
    model.train()
    
    # --- MONKEY-PATCH and ADD OPTIMIZER ---
    optimizer = AdamW(model.parameters()) # <-- CREATE OPTIMIZER
    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    random_dataset = np.random.randint(0, 10000, size=(10000,))
    warmup_steps = 5
    measure_steps = 10

    # --- Warmup Loop ---
    for _ in range(warmup_steps):
        inputs, targets = get_batch(random_dataset, 4, 512, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # --- Measurement Loop ---
    with nvtx.range("full_training_step"):
        for _ in range(measure_steps):
            inputs, targets = get_batch(random_dataset, 4, 512, device)
            with nvtx.range("forward_pass"):
                logits = model(inputs)
            with nvtx.range("backward_pass"):
                loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
            # --- ADD OPTIMIZER STEP WITH NVTX RANGE ---
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    profile_optimizer_step()