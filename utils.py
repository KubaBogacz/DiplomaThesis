import numpy as np
import scipy.signal as sp_sig
import re
import torch
import torch.nn as nn


def convert_datetime_to_time(datetime: np.ndarray, multi_day: bool = True) -> (np.ndarray, float):
    """
    Converts timestamp vector to time in seconds and estimates the sampling frequency.

    This function converts the input vector of timestamps (DateTime standard for the ICM+ data collection protocol)
    to time in seconds, with the recording starting at time 0, and estimates the sampling frequency based
    on the time difference between consecutive samples.

    Args:
        datetime (numpy array): The one-dimensional vector of timestamps.
        multi_day (bool): The flag which describes if the recording spans multiple day (multi_day = True) or not
            (multi_day = False). If not set to True, the time vector for recordings spanning multiple days may be
            incorrectly calculated at points when the date changes.

    Returns:
        t_hat (numpy array): Estimated time vector.
        fs_hat (float): Estimated sampling frequency (in Hz).
    """

    if not multi_day:
        t0 = (datetime[0] - np.floor(datetime[0])) * 24 * 3600
        t_hat = np.squeeze((datetime - np.floor(datetime)) * 24 * 3600 - t0)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
    else:
        n_datetime = datetime - datetime[0]
        n_datetime_days = np.floor(n_datetime)
        c_datetime = n_datetime - n_datetime_days
        c_datetime_seconds = c_datetime * 24 * 3600

        t_hat = []
        for idx in range(0, len(datetime)):
            c_t = n_datetime_days[idx] * 24 * 3600 + c_datetime_seconds[idx]
            t_hat.append(c_t)
        t_hat = np.asarray(t_hat)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
    return t_hat, fs_hat
 
 
def extract_recording_number(file_path):
    match = re.search(r'OCH_(\d+)_CLEAN_LR\.csv', file_path)
    return int(match.group(1)) if match else float('inf')


class CostNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def compute_all_costs(series1, series2, drop_cost_type="constant", drop_multiplier=1.0, percentile=95, cost_network=CostNetwork):
    """Compute alignment costs between time series with outlier handling"""
    # Validate inputs
    if not isinstance(series1, np.ndarray) or not isinstance(series2, np.ndarray):
        series1, series2 - np.array(series1), np.array(series2)
    
    # Compute pairwise cost    
    zx_costs = np.zeros((len(series1), len(series2)))
    for i in range(len(series1)):
        for j in range(len(series2)):
            zx_costs[i, j] = np.sqrt(np.sum((series1[i] - series2[j])**2))

    # Compute drop costs based on strategy
    if drop_cost_type == "constant":
        x_drop_costs = np.ones(len(series1)) * drop_multiplier
        z_drop_costs = np.ones(len(series2)) * drop_multiplier
        
    elif drop_cost_type == "percentile":
        threshold = np.percentile(zx_costs, percentile)
        x_drop_costs = np.ones(len(series1)) * threshold * drop_multiplier
        z_drop_costs = np.ones(len(series2)) * threshold * drop_multiplier
        
        
    elif drop_cost_type == "learnable":
        if cost_network is None:
            raise ValueError("must provide cost_network for learnable drop costs")
        
        mean1 = torch.tensor(np.mean(series1, axis=0), dtype=torch.float32)
        mean2 = torch.tensor(np.mean(series1, axis=0), dtype=torch.float32)
        
        with torch.no_grad():
            x_drop_costs = cost_network(mean2).item() * np.ones(len(series1))
            z_drop_costs = cost_network(mean1).item() * np.ones(len(series2))
            
    else:
        raise ValueError(f"Unknown drop_cost_type: {drop_cost_type}")
    
    # Copute drop probabilities
    # drop_probs = 1.0 / (1.0 + np.exp(-zx_costs + drop_costs))
    
    return zx_costs, x_drop_costs, z_drop_costs


def drop_dtw(costs, drop_costs, exclusive=True, contiguous=True, return_labels=False):
    """DTW algorithm allowing drops from one sequence.
    
    Parameters
    ----------
    costs: np.ndarray [N, M]
        Pairwise costs between points in sequences of length N and M
    drop_costs: np.ndarray [M] 
        Cost of dropping each point from second sequence
    exclusive: bool
        If True, point can only be matched with one point from other sequence
    contiguous: bool
        If True, can only match contiguous subsequences
    return_labels: bool
        If True, returns alignment labels instead of paths
    """
    N, M = costs.shape
    
    # Initialize solution matrices
    D = np.zeros([N + 1, M + 1, 2])  # States: 0=matched, 1=dropped
    D[1:, 0, :] = np.inf
    D[0, 1:, 0] = np.inf
    D[0, 1:, 1] = np.cumsum(drop_costs)

    # Initialize path tracking
    P = np.zeros([N + 1, M + 1, 2, 3], dtype=int)
    for i in range(1, M + 1):
        P[0, i, 1] = 0, i - 1, 1
        
    # Main DTW loop
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Get costs for matched state
            match_costs = [
                D[i-1, j-1, 0] + costs[i-1, j-1],  # diagonal
                D[i-1, j-1, 1] + costs[i-1, j-1],  # diagonal from dropped
            ]
            if not exclusive:
                match_costs.append(D[i-1, j, 0] + costs[i-1, j-1])  # vertical
            if not contiguous:
                match_costs.append(D[i, j-1, 0] + costs[i-1, j-1])  # horizontal
                
            # Update matched state
            best_match = np.argmin(match_costs)
            D[i, j, 0] = match_costs[best_match]
            
            # Update dropped state
            drop_costs_j = [
                D[i, j-1, 0] + drop_costs[j-1],
                D[i, j-1, 1] + drop_costs[j-1]
            ]
            best_drop = np.argmin(drop_costs_j)
            D[i, j, 1] = drop_costs_j[best_drop]
            
            # Track paths
            if best_match == 0:
                P[i, j, 0] = i-1, j-1, 0
            elif best_match == 1:
                P[i, j, 0] = i-1, j-1, 1
            else:
                P[i, j, 0] = i-1, j, 0 if best_match == 2 else i, j-1, 0
                
            P[i, j, 1] = i, j-1, best_drop

    # Backtrack solution
    final_state = np.argmin(D[N, M])
    min_cost = D[N, M, final_state]
    
    if return_labels:
        labels = np.zeros(M)
        i, j = N, M
        state = final_state
        while i > 0 or j > 0:
            if state == 0:
                labels[j-1] = i
            i_prev, j_prev, state = P[i, j, state]
            i, j = i_prev, j_prev
        return labels
    
    # Get path and dropped points
    path = []
    dropped = []
    i, j = N, M
    state = final_state
    while i > 0 or j > 0:
        path.append((i, j))
        if state == 1:
            dropped.append(j)
        i_prev, j_prev, state = P[i, j, state]
        i, j = i_prev, j_prev
        
    return min_cost, path, dropped


def double_drop_dtw(costs, drop_costs1, drop_costs2, contiguous=True):
    """DTW algorithm allowing drops from both sequences.
    
    Parameters
    ----------
    costs: np.ndarray [N, M]
        Pairwise costs between sequences
    drop_costs1: np.ndarray [N]
        Drop costs for first sequence
    drop_costs2: np.ndarray [M]
        Drop costs for second sequence
    contiguous: bool
        If True, only contiguous matches allowed
    """
    N, M = costs.shape
    INF = 1e9

    # Initialize tables: states are [matched, drop2, drop1, both_dropped]
    D = np.zeros([N + 1, M + 1, 4])
    D[1:, 0, :] = INF  
    D[0, 1:, :] = INF
    D[0, 0, 1:] = INF
    
    # Allow initial drops
    D[0, 1:, 1], D[0, 1:, 3] = np.cumsum(drop_costs2), np.cumsum(drop_costs2)
    D[1:, 0, 2], D[1:, 0, 3] = np.cumsum(drop_costs1), np.cumsum(drop_costs1)

    # Path tracking
    P = np.zeros([N + 1, M + 1, 4, 3], dtype=int)
    for i in range(1, N + 1):
        P[i, 0, 2] = [i-1, 0, 2]
        P[i, 0, 3] = [i-1, 0, 3]
    for j in range(1, M + 1):
        P[0, j, 1] = [0, j-1, 1]
        P[0, j, 3] = [0, j-1, 3]

    # Fill tables
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # State 0: Both matched
            match_costs = [
                D[i-1, j-1, 0] + costs[i-1, j-1],  # from matched
                D[i-1, j-1, 1] + costs[i-1, j-1],  # from drop2 
                D[i-1, j-1, 2] + costs[i-1, j-1],  # from drop1
                D[i-1, j-1, 3] + costs[i-1, j-1]   # from both dropped
            ]
            D[i, j, 0] = np.min(match_costs)
            P[i, j, 0] = [i-1, j-1, np.argmin(match_costs)]

            # State 1: Drop sequence 2
            drop2_costs = [
                D[i, j-1, 0] + drop_costs2[j-1],
                D[i, j-1, 1] + drop_costs2[j-1]
            ]
            D[i, j, 1] = np.min(drop2_costs)
            P[i, j, 1] = [i, j-1, np.argmin(drop2_costs)]

            # State 2: Drop sequence 1  
            drop1_costs = [
                D[i-1, j, 0] + drop_costs1[i-1],
                D[i-1, j, 2] + drop_costs1[i-1]
            ]
            D[i, j, 2] = np.min(drop1_costs)
            P[i, j, 2] = [i-1, j, np.argmin(drop1_costs)]

            # State 3: Drop both
            both_costs = [
                D[i-1, j, 1] + drop_costs1[i-1],
                D[i, j-1, 2] + drop_costs2[j-1],
                D[i-1, j-1, 3] + drop_costs1[i-1] + drop_costs2[j-1]
            ]
            D[i, j, 3] = np.min(both_costs)
            P[i, j, 3] = [i-1, j, 1] if np.argmin(both_costs) == 0 else \
                         [i, j-1, 2] if np.argmin(both_costs) == 1 else \
                         [i-1, j-1, 3]

    # Backtracking
    cur_state = np.argmin(D[N, M, :])
    min_cost = D[N, M, cur_state]

    path = []
    i, j = N, M
    dropped1 = [N] if cur_state in [2, 3] else []
    dropped2 = [M] if cur_state in [1, 3] else []
    
    while not (i == 0 and j == 0):
        path.append((i, j))
        i_prev, j_prev, prev_state = P[i, j, cur_state]
        if prev_state in [1, 3] and j > 0:
            dropped2.append(j)
        if prev_state in [2, 3] and i > 0:
            dropped1.append(i)
        i, j, cur_state = i_prev, j_prev, prev_state

    path.reverse()  # Reverse the path to start from the beginning

    return min_cost, path, dropped1, dropped2