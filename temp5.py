import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import emd
from dtw import *
from utils import *
from plotters import *

# Set the random seed for reproducibility
np.random.seed(42)

# Generate signal containing outliers
time = np.linspace(0, 10, 10000)
contaminated_signal = np.sin(time)
num_outliers = 50 # 5th bottom percentile of outliers
outliers_indices = np.random.choice(len(time), num_outliers, replace=False)
contaminated_signal[outliers_indices] += np.random.normal(0, 1, num_outliers)

# Generate clean, shifted signal
clean_signal = np.sin(time*1.5 + np.pi / 2)

zx_costs, x_drop_costs, z_drop_costs = compute_all_costs(
    series1=clean_signal,
    series2=contaminated_signal,
    drop_cost_type="percentile",
    percentile=95
)
print("ZX Costs:", zx_costs)
print("X Drop Costs:", x_drop_costs)
print("Details about ZX Costs:", type(zx_costs), zx_costs.shape)
print("Details about X Drop Costs:", type(x_drop_costs), x_drop_costs.shape)
plt.show()