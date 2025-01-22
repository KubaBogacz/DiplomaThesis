import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import emd
import torch
import torch.nn as nn
from dtw import dtw, sakoeChibaWindow
from utils import compute_all_costs, double_drop_dtw
from plotters import dropDtwPlotTwoWay, dropDtwPlotThreeWay

if __name__ == "__main__":
    np.random.seed(42)

    # Generate signals
    time = np.linspace(0, 25, 1000)
    contaminated_signal = np.sin(time / 2)
    num_outliers = 50
    outliers_indices = np.random.choice(len(time), num_outliers, replace=False)
    contaminated_signal[outliers_indices] += np.random.normal(0, 1, num_outliers)
    clean_signal = np.sin(time + np.pi / 2)

    # Compute costs with learnable drop function
    zx_costs, x_drop_costs, z_drop_costs = compute_all_costs(
        series1=clean_signal,
        series2=contaminated_signal,
        drop_cost_type="learnable"
    )

    min_cost, matched_indices, dropped1, dropped2 = double_drop_dtw(
        costs=zx_costs,
        drop_costs1=x_drop_costs,
        drop_costs2=z_drop_costs,
        contiguous=True
    )

    # Plots
    dropDtwPlotTwoWay(
        xts=clean_signal,
        yts=contaminated_signal,
        xlab="Time [s]",
        offset=1.5,
        matched_indices=matched_indices,
        dropped1=dropped1,
        dropped2=dropped2
    )

    dropDtwPlotThreeWay(
        xts=clean_signal,
        yts=contaminated_signal,
        xlab="Index of clean",
        ylab="Index of contaminated",
        matched_indices=matched_indices,
        dropped1=dropped1,
        dropped2=dropped2
    )

    plt.show()