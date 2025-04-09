import yaml
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# %matplotlib widget


def load_config(config_file="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series
import numpy
from numba import njit


@njit()
def _njit_subsequence_path(acc_cost_mat, idx_path_end):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`
    """
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


@njit()
def _njit_local_squared_dist(x, y):
    """Compute the squared distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.

    Returns
    -------
    dist : float
        Squared distance between x and y.
    """
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff * diff
    return dist


@njit()
def _njit_local_custom_dist(x, y, alpha=1.0):
    """Compute the distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.

    Returns
    -------
    dist : float
        Squared distance between x and y.
    """
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff**alpha
    return dist


@njit()
def _njit_subsequence_cost_matrix(subseq, longseq, warp_penalty=0.0, alpha=1.0):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d)
        Subsequence time series.
    longseq : array-like, shape=(sz2, d)
        Reference time series.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            d = _njit_local_custom_dist(subseq[i], longseq[j], alpha=alpha)
            diag = cum_sum[i, j]
            vert = cum_sum[i, j + 1] + warp_penalty
            hori = cum_sum[i + 1, j] + warp_penalty
            cum_sum[i + 1, j + 1] = d + min(diag, vert, hori)
    return cum_sum[1:, 1:]


def m_dtw_subsequence_path(subseq, longseq, be=None, warp_penalty=0.0, alpha=1.0):
    r"""Compute sub-sequence Dynamic Time Warping (DTW) similarity measure
    between a (possibly multidimensional) query and a long time series and
    return both the path and the similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Compared to traditional DTW, here, border constraints on admissible paths
    :math:`\pi` are relaxed such that :math:`\pi_0 = (0, ?)` and
    :math:`\pi_L = (N-1, ?)` where :math:`L` is the length of the considered
    path and :math:`N` is the length of the subsequence time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. This implementation finds the best matching starting
    and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d) or (sz1,)
        A query time series.
        If shape is (sz1,), the time series is assumed to be univariate.
    longseq : array-like, shape=(sz2, d) or (sz2,)
        A reference (supposed to be longer than `subseq`) time series.
        If shape is (sz2,), the time series is assumed to be univariate.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`.
    float
        Similarity score
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, be=be)
    longseq = to_time_series(longseq, be=be)
    acc_cost_mat = _njit_subsequence_cost_matrix(
        subseq=subseq, longseq=longseq, warp_penalty=warp_penalty, alpha=alpha
    )
    global_optimal_match = be.argmin(acc_cost_mat[-1, :])
    path = _njit_subsequence_path(acc_cost_mat, global_optimal_match)
    return path, be.sqrt(acc_cost_mat[-1, :][global_optimal_match])


# Read config
config = load_config()

# Paths and optional file references
twix_file = config["data"]["twix_file"]
dicom_folder = config["data"]["dicom_folder"]
resp_file = config["data"].get("resp_file", None)

# Read TWIX, extract raw k-space, and derive sampling frequency
scans = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)

kspace = di.extract_image_data(scans)

framerate, frametime = di.get_dicom_framerate(dicom_folder)
n_phase_encodes_per_frame = kspace.shape[0] // config["data"]["n_frames"]
fs = framerate * n_phase_encodes_per_frame  # ECG / respiration sampling freq

# Load and detect peaks/troughs
resp_data = np.loadtxt(resp_file, skiprows=1, usecols=1)
# Resample to match the total number of k-space time points
resp_data = signal.resample(resp_data, kspace.shape[0])[:, np.newaxis]

resp_peaks = ecg_resp.detect_resp_peaks(
    resp_data, fs, method="scipy", height=0.6, prominence=0.15
)
resp_troughs = ecg_resp.detect_resp_peaks(
    -resp_data, fs, method="scipy", height=0.6, prominence=0.15
)

# Assume resp_data is your (N, 1) ndarray:
resp_data = resp_data.flatten()

# Define two full respiratory cycles as the template
cycle1_start, cycle1_end = 817, 1853
cycle2_start, cycle2_end = 1853, 2719

template = resp_data[cycle1_start:cycle2_end]  # Two full cycles
template = gaussian_filter1d(template, sigma=10)
template_norm = (template - np.min(template)) / (np.max(template) - np.min(template))

# Normalize the full respiratory signal
resp_data_norm = (resp_data - np.min(resp_data)) / (
    np.max(resp_data) - np.min(resp_data)
)

# Parameters
window_size = (cycle2_end - cycle1_start) // 2  # Length of one full cycle

# Store estimated phase at each time step
phases = np.full_like(resp_data_norm, np.nan, dtype=float)

# Loop through the signal in real-time fashion
for t in tqdm(range(window_size, len(resp_data_norm))):
    current_data = resp_data_norm[: t + 1]
    current_data = (current_data - np.min(current_data)) / (
        np.max(current_data) - np.min(current_data)
    )
    current_window = current_data[
        t - window_size + 1 :
    ]  # Most recent full cycle window
    current_window = gaussian_filter1d(current_window, sigma=10)

    # Subsequence DTW: match recent window to the 2-cycle template
    path, _ = m_dtw_subsequence_path(current_window, template_norm, warp_penalty=1)
    window_indices, template_indices = zip(*path)

    # Find where the last sample of the current window (i.e., t) aligns in the template
    if window_size - 1 in window_indices:
        idx_in_path = window_indices.index(window_size - 1)
        matched_template_index = template_indices[idx_in_path]

        # --- Phase calculation for two‑cycle template ---
        cycle_len = cycle1_end - cycle1_start  # assume both cycles equal length
        if matched_template_index < cycle_len:
            # first cycle
            phase_percent = matched_template_index / cycle_len
        else:
            # second cycle
            phase_percent = (matched_template_index - cycle_len) / cycle_len
        phases[t] = phase_percent
