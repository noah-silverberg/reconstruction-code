import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter1d
from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series
from numba import njit
import yaml
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import scipy.signal as signal

# ---------------------------------------------------------
# DTW functions with warp penalty and alpha exponentiation
# ---------------------------------------------------------


@njit()
def _njit_subsequence_path(acc_cost_mat, idx_path_end):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


@njit()
def _njit_local_custom_dist(x, y, alpha=2.0):
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff**alpha
    return dist


@njit()
def _njit_subsequence_cost_matrix(subseq, longseq, warp_penalty=0.0, alpha=2.0):
    l1, l2 = subseq.shape[0], longseq.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            d = _njit_local_custom_dist(subseq[i], longseq[j], alpha=alpha)
            diag = cum_sum[i, j]
            vert = cum_sum[i, j + 1] + warp_penalty
            hori = cum_sum[i + 1, j] + warp_penalty
            cum_sum[i + 1, j + 1] = d + min(diag, vert, hori)
    return cum_sum[1:, 1:]


def m_dtw_subsequence_path(subseq, longseq, be=None, warp_penalty=0.0, alpha=2.0):
    """Sub‑sequence DTW with warp penalty & power‑law distance.
    Returns the warping *path* and the *raw accumulated cost* (no sqrt).
    A finite cost is guaranteed (inf if no valid path)."""
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, be=be)
    longseq = to_time_series(longseq, be=be)

    acc_cost_mat = _njit_subsequence_cost_matrix(
        subseq=subseq, longseq=longseq, warp_penalty=warp_penalty, alpha=alpha
    )

    # ── pick best end‑column ───────────────────────────────────────────────────
    j_best = be.argmin(acc_cost_mat[-1, :])
    best_cost = acc_cost_mat[-1, j_best]
    if not be.isfinite(best_cost):  # no admissible path → return inf cost
        return [], be.inf

    path = _njit_subsequence_path(acc_cost_mat, j_best)
    return path, best_cost


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------


def ground_truth_phases(length, peaks, troughs):
    extrema = np.sort(np.concatenate((peaks, troughs)))
    extrema = extrema[(extrema > 0) & (extrema < length - 1)]
    phases_gt = np.full(length, np.nan)
    for i in range(len(extrema) - 1):
        start, end = extrema[i], extrema[i + 1]
        if start in troughs and end in peaks:
            for t in range(start, end):
                phases_gt[t] = (t - start) / (end - start)
        elif start in peaks and end in troughs:
            for t in range(start, end):
                phases_gt[t] = 1 - (t - start) / (end - start)
    return phases_gt


def score_params(
    warp_penalty,
    alpha,
    resp_data_norm,
    template_norm,
    peaks,
    troughs,
    cycle_len,
    window_size,
):
    phases = np.full_like(resp_data_norm, np.nan, dtype=float)
    for t in range(window_size, len(resp_data_norm)):
        window = resp_data_norm[t - window_size + 1 : t + 1]
        window = gaussian_filter1d(window, sigma=10)
        path, _ = m_dtw_subsequence_path(
            window, template_norm, warp_penalty=warp_penalty, alpha=alpha
        )
        if len(path) == 0:
            continue  # no admissible alignment
            # --- use the exact mapping logic from the working notebook ---
        win_idx, temp_idx = zip(*path)
        if window_size - 1 not in win_idx:
            continue  # alignment does not include newest sample
        idx_in_path = win_idx.index(window_size - 1)
        matched_template_index = temp_idx[idx_in_path]

        # 0 % at trough → 100 % at peak → 0 % at next trough
        if matched_template_index < cycle_len:
            phase = matched_template_index / cycle_len  # rising
        else:
            phase = 1 - (matched_template_index - cycle_len) / cycle_len  # falling
        phases[t] = phase
    phases_gt = ground_truth_phases(len(resp_data_norm), peaks, troughs)
    valid = ~np.isnan(phases_gt) & ~np.isnan(phases)
    if np.sum(valid) == 0:
        return warp_penalty, alpha, np.inf, None
    mae = np.mean(np.abs(phases[valid] - phases_gt[valid])) if np.sum(valid) else np.inf
    return warp_penalty, alpha, mae, phases


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    scans = di.read_twix_file(
        config["data"]["twix_file"], include_scans=[-1], parse_pmu=False
    )
    kspace = di.extract_image_data(scans)
    framerate, _ = di.get_dicom_framerate(config["data"]["dicom_folder"])
    fs = framerate * (kspace.shape[0] // config["data"]["n_frames"])

    resp_data = np.loadtxt(config["data"]["resp_file"], skiprows=1, usecols=1)
    resp_data = signal.resample(resp_data, kspace.shape[0])[:, np.newaxis]
    resp_peaks = ecg_resp.detect_resp_peaks(
        resp_data, fs, method="scipy", height=0.6, prominence=0.15
    )
    resp_troughs = ecg_resp.detect_resp_peaks(
        -resp_data, fs, method="scipy", height=0.6, prominence=0.15
    )

    resp_data = resp_data.flatten()
    template = resp_data[817:2719]
    template = gaussian_filter1d(template, sigma=10)
    template_norm = (template - np.min(template)) / (
        np.max(template) - np.min(template)
    )
    resp_data_norm = (resp_data - np.min(resp_data)) / (
        np.max(resp_data) - np.min(resp_data)
    )

    # Show ground truth phase overlay before anything else
    phases_gt = ground_truth_phases(len(resp_data), resp_peaks, resp_troughs)
    plt.figure(figsize=(12, 4))
    plt.plot(resp_data_norm, label="Normalized Respiratory Signal")
    plt.plot(phases_gt, label="Ground Truth Phase", linestyle="--")
    plt.xlabel("Time (samples)")
    plt.ylabel("Normalized Amplitude / Phase")
    plt.legend()
    plt.title("Ground Truth Phase Overlay")
    plt.tight_layout()
    plt.show()

    cycle_len = 1853 - 817
    window_size = cycle_len

    warp_penalties = np.array([1.0])
    alphas = np.array([2.0])
    param_grid = list(itertools.product(warp_penalties, alphas))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                score_params,
                wp,
                a,
                resp_data_norm,
                template_norm,
                resp_peaks,
                resp_troughs,
                cycle_len,
                window_size,
            )
            for wp, a in param_grid
        ]
        for f in as_completed(futures):
            print(f"Completed {len(results) + 1}/{len(param_grid)}")
            results.append(f.result())

    score_matrix = np.full((len(warp_penalties), len(alphas)), np.nan)
    for wp, a, score, _ in results:
        i = np.where(np.isclose(warp_penalties, wp))[0][0]
        j = np.where(np.isclose(alphas, a))[0][0]
        score_matrix[i, j] = score

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        score_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=np.round(alphas, 2),
        yticklabels=np.round(warp_penalties, 2),
    )
    plt.xlabel("alpha")
    plt.ylabel("warp_penalty")
    plt.title("Mean Absolute Error of Phase Estimate")
    plt.tight_layout()
    plt.savefig("dtw_param_grid_heatmap.png", dpi=200)
    plt.show()

    # Plot best up to 3 parameter results
    sorted_results = sorted(results, key=lambda x: x[2])[:3]
    fig, axes = plt.subplots(
        len(sorted_results), 1, figsize=(12, 4 * len(sorted_results))
    )
    if len(sorted_results) == 1:
        axes = [axes]
    for ax, (wp, a, _, phase_est) in zip(axes, sorted_results):
        ax.plot(resp_data_norm, label="Normalized Respiratory Signal")
        ax.plot(phases_gt, label="Ground Truth", linestyle="--")
        ax.plot(phase_est, label=f"Estimated (wp={wp}, alpha={a})")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Phase (%)")
        ax.set_title(f"Phase Estimation Comparison: wp={wp}, alpha={a}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("dtw_phase_estimates_top3.png", dpi=200)
    plt.show()
