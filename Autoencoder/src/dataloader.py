import torch

import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
from arff2pandas import a2p

def load_data():
    with open('../data/ECG5000_TRAIN.arff') as f:
        train = a2p.load(f)

    with open('../data/ECG5000_TEST.arff') as f:
        test = a2p.load(f)

    # Conbine data for autoencoder and shuffle
    df = train.append(test)
    df = df.sample(frac=1.0)

    # Change reference name for convenience
    new_columns = list(df.columns)
    new_columns[-1] = 'target'
    df.columns = new_columns

    return df

def data_preprocess(df):
    normal_df = df[df.target == str(1)].drop(labels='target', axis=1)
    anomaly_df = df[df.target != str(1)].drop(labels='target', axis=1)

    train_df, val_df = train_test_split(
      normal_df,
      test_size=0.2,
    )

    val_df, test_df = train_test_split(
      val_df,
      test_size=0.3,
    )

    return train_df, val_df, test_df

def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features
