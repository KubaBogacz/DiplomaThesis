import os
import matplotlib.pyplot as plt
import emd
from dtw import *
from utils import *
import polars as pl
import numpy as np
from datetime import datetime
from utils import double_drop_dtw_1d


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

# Apply double_drop_dtw with dynamic drop costs
min_cost, D, P, best_state = double_drop_dtw_1d(imf_hr, imf_mx, percentile=50)
print(f"Double-drop DTW distance for series {file_number}: {min_cost}")

# Compare with standard DTW
alignment = dtw(imf_hr, imf_mx, keep_internals=True)
print(f"Standard DTW distance for series {file_number}: {alignment.distance}")

# Get alignment path
def get_alignment_path(P, best_state, K, N):
    path = []
    current = (K, N, best_state)
    while current[0] > 0 or current[1] > 0:
        path.append((current[0]-1, current[1]-1))  # -1 to convert to 0-based indices
        current = tuple(P[current])
    return np.array(path[::-1])  # Reverse path

# Get alignment indices
path = get_alignment_path(P, best_state, len(imf_mx), len(imf_hr))
index1, index2 = path[:, 1], path[:, 0]

# Create DTW object for compatibility with customDtwPlotTwoWay
class DtwResult:
    def __init__(self, index1, index2, query, reference):
        self.index1 = index1
        self.index2 = index2
        self.query = query
        self.reference = reference

dtw_result = DtwResult(index1, index2, imf_hr, imf_mx)

# Plot alignment using customDtwPlotTwoWay
plt.figure(figsize=(15, 10))
ax = customDtwPlotTwoWay(dtw_result, offset=2, 
                         xlab='Time (ms)', 
                         ylab='Normalized amplitude',
                         match_col='blue',
                         match_indices=50)  # Reduce number of shown matches for clarity
plt.title(f'Double-drop DTW Alignment (cost: {min_cost:.2f})')
plt.tight_layout()

# Visualize alignments
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(imf_hr, label='HR IMF')
plt.plot(imf_mx, label='MX IMF')
plt.title('Original Signals')
plt.legend()

plt.subplot(122)
plt.imshow(D[:,:,0], aspect='auto', origin='lower')
plt.colorbar(label='DTW Cost')
plt.title('DTW Cost Matrix')
plt.tight_layout()
plt.show()
