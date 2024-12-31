import os
import matplotlib.pyplot as plt
import emd
from dtw import *
from utils import *
import polars as pl
import numpy as np

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

# Create directory for plots
plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# Loop through files and save plots
# for file_number in range(len(file_paths)):
#     imf_mx = emd.sift.sift(mx[file_number])
#     n = imf_mx.shape[1]
#     imf_mx = imf_mx[:, n - 3] + imf_mx[:, n - 2] + imf_mx[:, n - 1]
#     imf_hr = emd.sift.sift(hr[file_number])
#     n = imf_hr.shape[1]
#     imf_hr = np.sum(imf_hr[:, 1:], axis=1)
#     imf_hr = (imf_hr - np.min(imf_hr)) / (np.max(imf_hr) - np.min(imf_hr))
#     imf_mx = (imf_mx - np.min(imf_mx)) / (np.max(imf_mx) - np.min(imf_mx))

#     alignment = dtw(imf_hr, imf_mx, keep_internals=True)
#     print(f"Alignment distance for series {file_number}: {alignment.distance}")
    
#     customDtwPlotTwoWay(alignment, offset=1.5, xlab="Time [s]")
    
#     plot_filename = os.path.join(plot_dir, f"plot_{file_number}.png")
#     plt.savefig(plot_filename)
#     plt.close()

# Save the first plot as a PDF
file_number = 6
print(f"Length of hr: {len(hr[6])}")
imf_mx = emd.sift.sift(mx[file_number])
n = imf_mx.shape[1]
imf_mx = imf_mx[:, n - 3] + imf_mx[:, n - 2] + imf_mx[:, n - 1]
imf_hr = emd.sift.sift(hr[file_number])
n = imf_hr.shape[1]
imf_hr = np.sum(imf_hr[:, 1:], axis=1)
imf_hr = (imf_hr - np.min(imf_hr)) / (np.max(imf_hr) - np.min(imf_hr))
imf_mx = (imf_mx - np.min(imf_mx)) / (np.max(imf_mx) - np.min(imf_mx))

alignment = dtw(imf_hr, imf_mx, keep_internals=True)
print(f"Alignment distance for series {file_number}: {alignment.distance}")

# alignment.plot(type="threeway", xlab="Index of HR", ylab="Index of Mx")
customDtwPlotTwoWay(alignment, offset=1.5, xlab="Time [s]")
plot_filename = os.path.join(r"E:\_PracaDyplomowa\plots\saved", f"plot_twoway_{file_number}.pdf")
plt.savefig(plot_filename, format='pdf')
plt.close()

plt.show()