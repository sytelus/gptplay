# from: https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/train_gpt2.py
# Modified Karpathy's llm.c python baseline with logging

# torchrun --nproc_per_node=1 --standalone kp_train_gpt2.py --batch_size 4 --num_iterations 6 --val_loss_every 2 --total_batch_size 4096
# python kp_train_gpt2.py --batch_size 4 --num_iterations 6 --val_loss_every 2 --total_batch_size 4096


"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
1 GPU:
python train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
you can also turn on flash-attention by appending --flash=1
4 GPU:
torchrun --standalone --nproc_per_node=4 train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
"""

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

from nanugpt.common import setup_logger


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1 # type: ignore
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1 # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # don't init this one, we will tie weights
        self.lm_head.LLMC_SKIP_INIT = 1 # type: ignore
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

class DistributedDataLoader:
    def __init__(self, filepath, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        print0(f"DataLoader: loading data from {filepath}")
        if ddp_world_size > 1:
            self.tokens = np.array(np.memmap(filepath, dtype=np.uint16, mode="r"))
        else:
            self.tokens = np.memmap(filepath, dtype=np.uint16, mode="r")
        print0(f"DataLoader: loaded {len(self.tokens):,} tokens")
        self.ntok_total = len(self.tokens)

        assert self.ntok_total > process_rank*B*T + B*T, f"dataset is too small for this process rank: {self.ntok_total} <= {process_rank*B*T + B*T + 1}, ntok_total={self.ntok_total}, process_rank={process_rank}, B={B}, T={T}"
        print0(f"DataLoader: total number of tokens: {self.ntok_total:,}")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
def pad_vocab(tensor, multiple=128, value=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    assert tensor.ndim == 2
    V, C = tensor.shape
    assert V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    assert padded.shape == (Vp, C)
    return padded

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"} # float16 todo maybe later
    version = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.size(0) # padded vocab size store in header
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, value=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}") # type: ignore

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="$DATA_ROOT/tokenized/openwebtext/tiktoken/train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="$DATA_ROOT/tokenized/openwebtext/tiktoken/val.bin", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="$OUT_DIR", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="d12", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=32, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=18865, help="number of iterations to run")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0006, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=700, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=0.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=250, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=200, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=0, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=1, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=1, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=1, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=1, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=0, help="write tensors to disk")

    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d-%H%M%S"), help="unique identifier for this run")

    args = parser.parse_args()

    run_id = args.run_id

    args.input_bin = os.path.expandvars(args.input_bin)
    args.input_val_bin = os.path.expandvars(args.input_val_bin)
    args.output_dir = os.path.join(os.path.expandvars(args.output_dir), run_id)

    logger = setup_logger(config={
            "logging":{
                    "project_name": os.getenv("JOB_NAME", "keller_train_gpt2"),
                    "run_name": run_id,
                    "enable_wandb": True,
                    "log_dir": args.output_dir,
                    "log_filename": "log.txt",
                    "summaries_filename": "summary.txt",
                    "allow_overwrite_log": True,
                    "metrics_type": "classification",
                    "summaries_stdout": True,
                },
    })

    # convert args to dict and log
    logger.info(vars(args))

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    ddp_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    ddp_rank = int(os.environ.get('RANK', '0'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        assert gpu_count-1 >= ddp_local_rank, f'LOCAL_RANK={ddp_local_rank} is greater than available GPUs={gpu_count}'
        torch.cuda.set_device(ddp_local_rank)
        device_name = f'cuda:{ddp_local_rank}'
        device_id = ddp_local_rank
    elif gpu_count == 1:
        torch.cuda.set_device(0)
        device_name = 'cuda:0'
        device_id = 0
    else:
        raise ValueError('No GPU found. Set device_type=cpu.')
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = ddp_world_size > 1
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        init_process_group(backend='nccl', device_id=device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        master_process = True
        seed_offset = 0
        zero_stage = 0

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    if master_process and args.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    else:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_to_size = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_to_size.update({f"d{d}": f"d{d}" for d in [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model] # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    torch.cuda.reset_peak_memory_stats()

    timings = []
    train_tokens = 0
    train_start_time = time.time()
    train_time_hr = 0.0
    metrics = {}
    norm = -1.0   # dummy value to print in inference-only mode
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        val_time = 0.0
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            torch.cuda.synchronize()
            val_start = time.time()
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
            val_loss /= args.val_max_steps
            if ddp:
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
            val_time = time.time() - val_start
            metrics.update({
                "train/step": step,
                "val/step": step,
                "val/loss": val_loss,
                "val/time_s": val_time,
            })
            torch.cuda.synchronize()

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            logger.info(metrics)
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item() # type: ignore
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        train_tokens += grad_accum_steps * ddp_world_size * B * T
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        train_time_hr += (t1-t0-val_time) / 3600.0
        metrics.update({
            "train/loss": lossf,
            "train/token_per_sec": tokens_per_second,
            "train/step_interval": t1-t0-val_time,
            "train/elapsed_hr": (t1-train_start_time) / 3600.0,
            "train/train_time_hr": train_time_hr,
            "train/tokens": train_tokens,
            "train/lr": lr,
            "train/pre_clip_norm": norm,
        })

        logger.info(metrics)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    logger.info({
        "final_iters_avg (usec)": np.mean(timings),
        "final_iters_stddev (usec)": np.std(timings),
        "pick_memory_consumption (MiB)": torch.cuda.max_memory_allocated() // 1024 // 1024,
    })

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()