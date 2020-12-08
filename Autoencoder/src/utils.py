import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from train_test import predict_lstm

def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
      path_deviation.index,
      under_line,
      over_line,
      alpha=.125
    )
    ax.set_title(class_name)

def plot_time_series_data(df, class_names):
    classes = df.target.unique()

    fig, axs = plt.subplots(
      nrows=len(classes) // 3 + 1,
      ncols=3,
      sharey=True,
      figsize=(14, 8)
    )

    for i, cls in enumerate(classes):
        ax = axs.flat[i]
        data = df[df.target == cls] \
        .drop(labels='target', axis=1) \
        .mean(axis=0) \
        .to_numpy()
        plot_time_series_class(data, class_names[i], ax)

    fig.delaxes(axs.flat[-1])
    fig.tight_layout();

def plot_prediction(data, model, criterion, title, ax):
    predictions, pred_losses = predict_lstm(model, [data], criterion)

    ax.plot(data, label='true')
    ax.plot(predictions[0], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
    ax.legend()
