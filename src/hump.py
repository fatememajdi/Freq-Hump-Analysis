import numpy as np
import pandas as pd


def detect_hump_range_clusters(signal, min_freq=500, max_freq=None, top_percent=15, gap_threshold=100):
    """
    Detects the most prominent frequency range (hump) in a signal using clustering of high-energy components.

    This method selects the top `top_percent` energy frequencies (after clipping high outliers)
    and groups them into clusters based on proximity (less than `gap_threshold` Hz).
    The largest cluster is selected as the hump range.

    Parameters:
    - signal (TS): signal object containing `fq` (frequency array) and `ft` (magnitude spectrum)
    - min_freq (int, optional): minimum frequency to consider for hump detection (default: 500 Hz)
    - max_freq (int or None, optional): maximum frequency to consider (default: last frequency in `fq`)
    - top_percent (int, optional): percentage of highest spectral magnitudes to include (default: 15)
    - gap_threshold (int, optional): maximum allowed gap (in Hz) between consecutive high-energy frequencies
                                     to consider them part of the same cluster (default: 100)

    Returns:
    - tuple: (start_freq, end_freq) of the detected hump range
    """
    fq = signal.fq
    ft = signal.ft

    min_freq = max(min_freq, fq[1])
    max_freq = max_freq or fq[-1]

    mask = (fq >= min_freq) & (fq <= max_freq)
    fq = fq[mask]
    ft = ft[mask]

    avg_val = np.mean(ft)
    std_val = np.std(ft)
    threshold_value = avg_val + (4 * std_val)
    ft_clipped = np.where(ft >= threshold_value, 0, ft)

    threshold_top = np.percentile(ft_clipped, 100 - top_percent)
    top_freqs = fq[ft_clipped >= threshold_top]

    if len(top_freqs) == 0:
        return (min_freq, min_freq + 500)

    sorted_freqs = np.sort(top_freqs)

    clusters = []
    cluster = [sorted_freqs[0]]

    for f in sorted_freqs[1:]:
        if f - cluster[-1] <= gap_threshold:
            cluster.append(f)
        else:
            clusters.append(cluster)
            cluster = [f]
    clusters.append(cluster)

    largest_cluster = max(clusters, key=len)

    hump_range = (int(largest_cluster[0]), int(largest_cluster[-1]))

    return hump_range


def normalize_and_clip(signal_ft, std_factor=4):
    """
    Normalizes and clips a frequency spectrum by zeroing out values above a statistical threshold.

    Parameters:
    - signal_ft (np.ndarray): 1D array of spectral magnitudes (usually ft from FFT)
    - std_factor (float, optional): number of standard deviations above mean to set as threshold (default: 4)

    Returns:
    - np.ndarray: clipped spectrum where values above (mean + std_factor * std) are set to zero
    """
    avg_val = np.mean(signal_ft)
    std_val = np.std(signal_ft)
    threshold = avg_val + std_factor * std_val
    clipped = np.where(signal_ft > threshold, 0, signal_ft)
    return clipped


def check_hump_with_iso_ratio(signal_obj, freq_range, iso_energy, std_factor=4, ratio_threshold=0.05):
    """
    Checks whether the energy of a given frequency range (potential hump) is significant
    compared to the total energy (above 500 Hz), scaled by a given ISO energy value.

    Parameters:
    - signal_obj (TS): a signal object containing frequency (fq) and spectrum (ft)
    - freq_range (tuple): frequency range (start_freq, end_freq) to check for hump
    - iso_energy (float): scaling factor based on ISO baseline energy
    - std_factor (float, optional): factor for standard deviation clipping during normalization (default: 4)
    - ratio_threshold (float, optional): minimum ratio threshold to consider as significant (default: 0.05)

    Returns:
    - (bool, float):
        - True if scaled energy ratio in the given range is above the threshold, False otherwise
        - Computed energy ratio
    """

    ft = np.array(signal_obj.ft)
    fq = np.array(signal_obj.fq)

    ft_clipped = normalize_and_clip(ft, std_factor=std_factor)

    mask_total = fq >= 500
    total_energy = np.sum(ft_clipped[mask_total] ** 2)

    mask_range = (fq >= freq_range[0]) & (fq < freq_range[1])
    range_energy = np.sum(ft_clipped[mask_range] ** 2)

    if total_energy == 0:
        return False

    ratio = ((range_energy / total_energy) * iso_energy)

    return ratio >= ratio_threshold, ratio


def extract_hump_center(hump_range):
    return (hump_range[0] + hump_range[1]) / 2 if hump_range else None


def extract_hump_center(hump_range):
    if pd.isna(hump_range):
        return None
    try:
        start, end = map(float, str(hump_range).split('-'))
        return (start + end) / 2
    except:
        return None


def detect_downward_shifts(df: pd.DataFrame, min_shift_hz=1000):
    df = df.copy()

    df['Timestamp'] = pd.to_datetime(df['DATE'].astype(
        str) + ' ' + df['TIME'].astype(str), errors='coerce')
    df["hump_center"] = df["hump_range"].apply(
        lambda r: (int(r.strip("()").split(",")[
                   0]) + int(r.strip("()").split(",")[1])) / 2
        if pd.notnull(r) and isinstance(r, str) else None
    )


    df['is_range_shift'] = False

    for i, row in df.iterrows():
        past_rows = df[
            (df['COMP_NUMBER'] == row['COMP_NUMBER']) &
            (df['ASSIGNMENT'] == row['ASSIGNMENT']) &
            (df['Measuring_Point'] == row['Measuring_Point']) &
            (df['Direction'] == row['Direction']) &
            (df['Timestamp'] < row['Timestamp']) &
            (~df['hump_center'].isna())
        ]

        if past_rows.empty or pd.isna(row['hump_center']):
            continue

        latest_row = past_rows.sort_values('Timestamp').iloc[-1]
        center_diff = row['hump_center'] - latest_row['hump_center']

        if center_diff < 0 and abs(center_diff) > min_shift_hz:
            df.at[i, 'is_range_shift'] = True
        else:
            df.at[i, 'is_range_shift'] = False

    return df
