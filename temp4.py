import os
import matplotlib.pyplot as plt
import emd
from dtw import *
from utils import *
import polars as pl
import numpy as np
from utils import *


# from dp.dp_utils import compute_all_costs

time = []
fs = []
hr = []
mx = []

# Recordings files paths
path = os.getcwd() + r"\data_healthy"
file_paths = sorted([os.path.join(path, file) for file in os.listdir(path)], key=extract_recording_number)

# Read data from the files
for file_path in file_paths:
    df = pl.read_csv(file_path, separator=";", decimal_comma=True)
    t_hat, _ = convert_datetime_to_time(df["DateTime"].to_numpy())
    time.append(t_hat)
    hr.append(df["hr"].to_numpy())
    if df["FVL"].mean() > df["FVR"].mean():
        mx.append(df["mx_l"].to_numpy())
    else:
        mx.append(df["mx_r"].to_numpy())
        
file_number = 13
imf_mx = emd.sift.sift(mx[file_number])
n = imf_mx.shape[1]
imf_mx = imf_mx[:, n - 3] + imf_mx[:, n - 2] + imf_mx[:, n - 1]
imf_hr = emd.sift.sift(hr[file_number])
n = imf_hr.shape[1]
imf_hr = np.sum(imf_hr[:, 1:], axis=1)
imf_hr = (imf_hr - np.min(imf_hr)) / (np.max(imf_hr) - np.min(imf_hr))
imf_mx = (imf_mx - np.min(imf_mx)) / (np.max(imf_mx) - np.min(imf_mx))


# Compute costs between signals
zx_costs, drop_costs, drop_probs = compute_all_costs(
    series1=imf_hr, 
    series2=imf_mx, 
    drop_cost_type="percentile", 
    percentile=90
)

print(drop_costs.shape)
print(drop_costs.size)
print(drop_costs.dtype)

print(zx_costs.shape)
print(zx_costs.size)
print(zx_costs.dtype)