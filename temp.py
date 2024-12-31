import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import emd
from dtw import *
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
        
# Each file has the same fs, thus there is no need to iterate over files
sample_rate = 1 / np.mean(np.diff(time[0]))
file_number = 15


# Standard min/max norm
# hr = [np.interp(h, (h.min(), h.max()), (0, 1)) for h in hr]
# mx = [np.interp(m, (m.min(), m.max()), (0, 1)) for m in mx]

# Z-score normalization
hr = [(h - np.mean(h)) / np.std(h) for h in hr]
mx = [(m - np.mean(m)) / np.std(m) for m in mx]

imf = emd.sift.sift(mx[file_number])
n = imf.shape[1]
imf_sum = imf[:, n - 3] + imf[:, n - 2] + imf[:, n - 1]
alignment = dtw(hr[file_number], imf_sum, keep_internals=True)
print(f"Alignment distance for series {file_number}: {alignment.distance}")

# customDtwPlotTwoWay(alignment, offset=2, xlab="Time [s]")
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time[file_number], mx[file_number])
plt.xticks([])
plt.ylabel('Raw')
yticks = plt.gca().get_yticks()
yticklabels = plt.gca().get_yticklabels()

plt.subplot(2, 1, 2)
plt.plot(time[file_number], imf_sum)
plt.yticks(yticks[1:-1], yticklabels[1:-1])
plt.xlabel('Time [s]')
plt.ylabel('EMD')

plt.show()
