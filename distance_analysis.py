import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest
import os

df = pd.read_csv('distances.csv')
columns = ['DTW Normalized Distances', 'DropDTW Normalized Distances']

# Print minimal and maximal values of normalized distances along with their indices
for col in columns:
    min_val = df[col].min()
    max_val = df[col].max()
    min_idx = df[col].idxmin()
    max_idx = df[col].idxmax()
    print(f"{col}: Min = {min_val} (Index = {min_idx}), Max = {max_val} (Index = {max_idx})")

# Calculate the mean difference in distance values between DropDTW and DTW
mean_difference = (df['DropDTW Normalized Distances'] - df['DTW Normalized Distances']).mean()
print(f"Mean difference between DropDTW and DTW Normalized Distances: {mean_difference}")

# Perform normality tests and print results
for col in columns:
    print(f"\nAnalyzing {col}:")
    
    # Shapiro-Wilk Test
    stat, p = shapiro(df[col])
    print(f"Shapiro-Wilk Test: W={stat:.4f}, p={p:.4e}")
    if p > 0.05:
        print(f"The data in '{col}' appears to be normally distributed (p > 0.05).")
    else:
        print(f"The data in '{col}' does not appear to be normally distributed (p <= 0.05).")
    
    # D'Agostino and Pearson Test
    stat, p = normaltest(df[col])
    print(f"D'Agostino and Pearson's Test: stat={stat:.4f}, p={p:.4e}")
    if p > 0.05:
        print(f"The data in '{col}' appears to be normally distributed (p > 0.05).")
    else:
        print(f"The data in '{col}' does not appear to be normally distributed (p <= 0.05).")

output_dir = 'D:/_PracaDyplomowa/plots/cost_bar_charts'
os.makedirs(output_dir, exist_ok=True)

# # Plot bar charts
# for col in columns:
#     plt.figure(figsize=(8, 4))
#     df[col].plot(kind='bar', color='skyblue', edgecolor='black')
#     plt.xlabel("Sample number")
#     plt.ylabel("Normalized distance")
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{col}_bar.pdf"))
#     plt.show()

# # Plot distributions
# for col in columns:
#     plt.figure(figsize=(8, 4))
#     df[col].plot(kind='hist', bins=34, color='skyblue', edgecolor='black', alpha=0.7)
#     plt.xlabel("Normalized distance")
#     plt.ylabel("Number of samples")
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{col}_histogram.pdf"))
#     plt.show()