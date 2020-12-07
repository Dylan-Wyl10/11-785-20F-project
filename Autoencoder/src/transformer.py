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
    def __init__(self, n_features, embedding_dim=64):
        super(TransformerAutoencoder, self).__init__()
        self.emb = nn.Linear(n_features, embedding_dim)
        self.trans = nn.Transformer( d_model=embedding_dim)
        self.out = nn.Linear(embedding_dim, n_features)
    def forward(self, x):
        #print(x.shape)
        x= x.unsqueeze(0)
        x = self.emb(x)
        x = x.permute(1,0,2)
        x = self.out(self.trans(x,x))
        x=x.squeeze(1)
        #print(x.shape)
        return x