import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal

import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series
from numba import njit

# -------------------------------------------------------------
# Utility: load YAML config
# -------------------------------------------------------------


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------
#  Low‑level DTW helpers (unchanged phase logic)
# -------------------------------------------------------------


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
                    acc_cost_mat[i - 1, j - 1],
                    acc_cost_mat[i - 1, j],
                    acc_cost_mat[i, j - 1],
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
def _njit_local_custom_dist(x, y, alpha=1.0):
    d = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        d += np.abs(diff) ** alpha
    return d


@njit()
def _njit_subsequence_cost_matrix(subseq, longseq, warp_penalty=0.0, alpha=1.0):
    l1, l2 = subseq.shape[0], longseq.shape[0]
    cum = np.full((l1 + 1, l2 + 1), np.inf)
    cum[0, :] = 0.0
    for i in range(l1):
        for j in range(l2):
            d = _njit_local_custom_dist(subseq[i], longseq[j], alpha)
            diag = cum[i, j]
            vert = cum[i, j + 1] + warp_penalty
            hori = cum[i + 1, j] + warp_penalty
            cum[i + 1, j + 1] = d + min(diag, vert, hori)
    return cum[1:, 1:]


def m_dtw_subsequence_path(subseq, longseq, warp_penalty=0.0, alpha=1.0, be=None):
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, be=be)
    longseq = to_time_series(longseq, be=be)
    acc = _njit_subsequence_cost_matrix(subseq, longseq, warp_penalty, alpha)
    j_best = be.argmin(acc[-1, :])
    path = _njit_subsequence_path(acc, j_best)
    return path, acc[-1, j_best]


# -------------------------------------------------------------
#   Ground‑truth phase 0 → 1 → 0 using trough/peak indices
# -------------------------------------------------------------


def ground_truth_phase(n_samples, peaks, troughs):
    """Return array of shape (n_samples,) with NaN outside cycles."""
    peaks = np.asarray(peaks, dtype=int)
    troughs = np.asarray(troughs, dtype=int)
    extrema = np.sort(np.concatenate((peaks, troughs)))
    valid = (extrema > 0) & (extrema < n_samples - 1)
    extrema = extrema[valid]
    phase = np.full(n_samples, np.nan, dtype=float)
    for i in range(len(extrema) - 1):
        s, e = extrema[i], extrema[i + 1]
        if s in troughs and e in peaks:
            phase[s:e] = np.linspace(0, 1, e - s, endpoint=False)
        elif s in peaks and e in troughs:
            phase[s:e] = np.linspace(1, 0, e - s, endpoint=False)
    return phase


# -------------------------------------------------------------
#   Parameter scoring (keeps existing percentage logic)
# -------------------------------------------------------------


def score_params(arg_tuple):
    (
        warp_penalty,
        alpha,
        resp_norm,
        template_norm,
        window_size,
        peaks,
        troughs,
    ) = arg_tuple
    phases_est = np.full_like(resp_norm, np.nan, dtype=float)
    for t in range(window_size, len(resp_norm)):
        window = resp_norm[t - window_size + 1 : t + 1]
        window = gaussian_filter1d(window, sigma=10)
        path, _ = m_dtw_subsequence_path(window, template_norm, warp_penalty, alpha)
        if len(path) == 0:
            continue
        win_idx, temp_idx = zip(*path)
        if window_size - 1 not in win_idx:
            continue
        matched_template_index = temp_idx[win_idx.index(window_size - 1)]

        # pre‑compute template extrema helpers
        extrema_sorted = np.sort(np.concatenate((peaks, troughs)))

        # Get the phase
        prev_idx = extrema_sorted[extrema_sorted < matched_template_index]
        next_idx = extrema_sorted[extrema_sorted > matched_template_index]
        if len(prev_idx) == 0 or len(next_idx) == 0:
            continue
        prev_idx = prev_idx[-1]
        next_idx = next_idx[0]
        if prev_idx in peaks and next_idx in troughs:
            # peak -> trough
            phase = 1 - (matched_template_index - prev_idx) / (next_idx - prev_idx)
        elif prev_idx in troughs and next_idx in peaks:
            # trough -> peak
            phase = (matched_template_index - prev_idx) / (next_idx - prev_idx)
        else:
            # should not happen
            phase = np.nan

        phases_est[t] = phase
    phases_gt = ground_truth_phase(len(resp_norm), peaks, troughs)
    valid = ~np.isnan(phases_gt) & ~np.isnan(phases_est)
    mae = (
        np.mean(np.abs(phases_est[valid] - phases_gt[valid])) if valid.any() else np.inf
    )
    mse = (
        np.mean((phases_est[valid] - phases_gt[valid]) ** 2) if valid.any() else np.inf
    )
    cust_err = (
        np.mean(np.abs(phases_est[valid] - phases_gt[valid]) ** 8)
        if valid.any()
        else np.inf
    )
    print(f"wp={warp_penalty:.2f}, alpha={alpha:.2f}")
    print(f"MAE={mae:.3f}, MSE={mse:.3f}, CustErr={cust_err:.3f}\n")
    return warp_penalty, alpha, mae, mse, cust_err, phases_est


# -------------------------------------------------------------
#   Main
# -------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    twix_file = cfg["data"]["twix_file"]
    dicom_folder = cfg["data"]["dicom_folder"]
    resp_file = cfg["data"]["resp_file"]

    # read data & sampling rate
    scans = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)
    fr, _ = di.get_dicom_framerate(dicom_folder)
    fs = fr * (kspace.shape[0] // cfg["data"]["n_frames"])

    resp = np.loadtxt(resp_file, skiprows=1, usecols=1)
    resp = signal.resample(resp, kspace.shape[0])[:, np.newaxis]

    # detect extrema
    peaks_global = ecg_resp.detect_resp_peaks(
        resp, fs, method="scipy", height=0.6, prominence=0.15
    )
    troughs_global = ecg_resp.detect_resp_peaks(
        -resp, fs, method="scipy", height=0.6, prominence=0.15
    )

    resp = resp.flatten()

    # build template (two cycles hard‑coded indices)
    cycle1_start, cycle1_end = peaks_global[0], peaks_global[1]
    cycle2_end = peaks_global[2]
    template = gaussian_filter1d(resp[cycle1_start:cycle2_end], sigma=10)
    template_norm = (template - template.min()) / (template.max() - template.min())

    resp_norm = (resp - resp.min()) / (resp.max() - resp.min())

    cycle_len = cycle1_end - cycle1_start
    window_size = cycle_len / 2  # one full cycle

    # ----- parameter grid -----
    # warp_penalties = np.arange(0.0, 2.0, 0.2)
    # alphas = np.arange(0.8, 10.0, 0.4)
    warp_penalties = np.array([0.00, 0.04, 1.0, 2.0])
    alphas = np.array([2, 4, 8, 12, 16])
    grid = list(itertools.product(warp_penalties, alphas))

    print(f"Evaluating {len(grid)} parameter combinations …")

    args_iter = [
        (
            wp,
            a,
            resp_norm,
            template_norm,
            cycle_len,
            peaks_global,
            troughs_global,
        )
        for wp, a in grid
    ]

    results = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(score_params, arg) for arg in args_iter]
        for f in as_completed(futs):
            results.append(f.result())
            print(f"completed {len(results)}/{len(grid)}")

    # reshape into matrix for heatmap
    score_mat = np.full((len(warp_penalties), len(alphas)), np.nan)
    for wp, a, mae, mse, cust_err, _ in results:
        i = np.where(np.isclose(warp_penalties, wp))[0][0]
        j = np.where(np.isclose(alphas, a))[0][0]
        score_mat[i, j] = cust_err

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        score_mat,
        annot=True,
        fmt=".3f",
        xticklabels=np.round(alphas, 2),
        yticklabels=np.round(warp_penalties, 2),
    )
    plt.xlabel("alpha")
    plt.ylabel("warp_penalty")
    plt.title("Custom Error between estimated phase and ground truth (0→1→0)")
    plt.tight_layout()
    plt.savefig("grid_mae_heatmap.png", dpi=200)
    plt.show()

    # plot best three
    best = sorted(results, key=lambda x: x[2])[:3]
    gt = ground_truth_phase(len(resp_norm), peaks_global, troughs_global)
    fig, axes = plt.subplots(len(best), 1, figsize=(12, 4 * len(best)))
    if len(best) == 1:
        axes = [axes]
    for ax, (wp, a, mae, mse, cust_err, ph_est) in zip(axes, best):
        ax.plot(resp_norm, label="Resp signal", alpha=0.4)
        ax.plot(gt, label="Ground truth", linestyle="--")
        ax.plot(ph_est, label=f"Est (wp={wp:.2f}, a={a:.2f}, Cust Err={cust_err:.3f})")
        ax.set_ylabel("Phase / Amplitude")
        ax.legend()
        ax.legend()
    axes[-1].set_xlabel("Samples")
    plt.tight_layout()
    plt.savefig("top3_phase_traces.png", dpi=200)
    plt.show()
