"""
GPT model
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ..builder import HEADS

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self,
                 n_embd=512,
                 n_head=8,
                 block_size=100,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        y = self.resid_drop(y)
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd=512,
                 n_head=8,
                 block_size=100,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@HEADS.register_module()
class BboxEncoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self,
                 n_layer=12,
                 n_head=8,
                 n_embd=512,
                 bbox_cord_dim=4,
                 bbox_max_num=512,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1):
        super(BboxEncoder, self).__init__()

        # input embedding stem
        # self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.bbox_embedding = nn.Sequential(
            nn.Linear(bbox_cord_dim, n_embd),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(embd_pdrop)
            )
        
        # transformer 
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, bbox_max_num, 
                    attn_pdrop, resid_pdrop) for _ in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_embd, bias=False)

        self.block_size = bbox_max_num
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", 
                    sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, xs):
        out = []
        for x in xs:
            bbox_num, bbox_dim = x.size()

            assert bbox_num <= self.block_size, "Cannot forward, model block size is exhausted."

            # if x.is_cuda:
            #     device = x.get_device()
            # else:
            #     device = 'cpu'

            # if bbox_num >= self.block_size:
            #     x = x[:,:self.block_size,:]
            # else:
                # zero_cat = torch.Tensor(
                #     np.zeros([b,self.block_size-bbox_num,bbox_dim]), 
                #     dtype=np.float32, 
                #     device=device)
                # x = torch.cat((x,zero_cat),dim=1)
            # forward the GPT model
            x = x.unsqueeze(0)
            bbox_repr = self.bbox_embedding(x)
            print("bbox_repr, ", bbox_repr.max(), bbox_repr.min(), bbox_repr.mean(), bbox_repr.std())
            #bbox_repr --> [b, box_num, 512]
            x = self.blocks(bbox_repr)
            x = self.ln_f(x)
            logits = self.head(x)
            print("logits, ", logits.max(), logits.min(), logits.mean(), logits.std())
            out.append(logits.squeeze(0))
        return out
