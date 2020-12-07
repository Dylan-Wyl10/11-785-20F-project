# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:58:53 2020

@author: Nicky
"""
import torch
import numpy as np
from torch import nn, optim

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(TransformerAutoencoder, self).__init__()
        self.trans = nn.Transformer(nheads=5, d_model=n_features)
        
    def forward(self, x):
        return self.trans(x,x)