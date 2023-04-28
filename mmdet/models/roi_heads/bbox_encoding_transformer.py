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


@HEADS.register_module()
class BboxEncoder(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self,
                 n_layer=12,
                 n_head=8,
                 n_embd=512,
                 bbox_cord_dim=4,
                 bbox_max_num=1024,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1):
        super(BboxEncoder, self).__init__()

        # input embedding stem
        # self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.bbox_emb_layer = nn.ModuleList()
        self.bbox_emb_layer.append(nn.Linear(bbox_cord_dim, n_embd))
        self.bbox_emb_layer.append(nn.Linear(n_embd, n_embd))
        self.bbox_emb_layer.append(nn.Dropout(embd_pdrop))
        
        # transformer 
        transformer_layer=nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head, 
            batch_first=True,
            dropout=attn_pdrop)

        self.encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layer,
            enable_nested_tensor=True, 
            mask_check=True)

        self.block_size = bbox_max_num

        logger.info("number of parameters: %e", 
                    sum(p.numel() for p in self.parameters()))

    def forward(self, xs):
        # Forming the batch. Since the length of each rois is not same.
        masks = []
        inputs = []
        bboxnum_per_batch = []
        for x in xs:
            bbox_num, bbox_dim = x.size()
            assert bbox_num <= self.block_size, \
                "Cannot forward, model block size is exhausted."

            x=x.unsqueeze(0) # (1,bbox_num,4)

            mask = x.new(1,self.block_size).float().zero_()
            pad = x.new(1,self.block_size-bbox_num, bbox_dim).float().zero_()

            x = torch.cat((x,pad),dim=1)
            mask[:,:bbox_num] = 1

            inputs.append(x)
            masks.append(mask)
            bboxnum_per_batch.append(bbox_num)
        
        input = torch.cat(inputs, dim = 0)
        mask = torch.cat(masks,dim=0)

        for i, emb_layer in enumerate(self.bbox_emb_layer):
            input = emb_layer(input)
            input = input * mask.unsqueeze(-1)
        
        # Sending the embedded bbox into transformer.
        logits = self.encoder(input, src_key_padding_mask=mask.bool())

        out = []
        for i in range(logits.size(0)):
            bbox_num = bboxnum_per_batch[i]
            out.append(logits[i,:bbox_num,:])

        return out
