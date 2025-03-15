import numpy as np
import utils.reconstruction as reconstruction


def bin_reconstructed_kspace_by_cardiac_phase_kernel(
    kpca,
    X_kpca,
    selection,
    frame_shape,
    orig_feature_dim,
    r_peaks_list,
    num_bins=30,
    n_phase_encodes_per_frame=112,
    extended_pe_lines=128,
):
    """
    Reconstruct the k-space data using only the specified kernel PCA components (without applying inverse FFT),
    reshape the reconstructed k-space into its full 2D format, and then bin the data by cardiac phase.

    Parameters:
      kpca (KernelPCA object): Fitted KernelPCA model with fit_inverse_transform=True.
      X_kpca (np.ndarray): Kernel PCA transformed data of shape (n_frames, n_components_total).
      selection (int or list): Either an integer or a list of kernel PCA component indices (0-indexed) to use.
      frame_shape (tuple): Shape of a single frame in the original domain (n_phase_encodes_per_frame, n_coils, n_freq).
      orig_feature_dim (int): Original number of features per frame (before converting complex to real).
      r_peaks_list (list of np.ndarray): List of detected R-peak indices per ECG channel.
      num_bins (int): Number of cardiac bins.
      n_phase_encodes_per_frame (int): Number of phase encodes per frame.
      extended_pe_lines (int): Number of phase encode lines after extension.

    Returns:
      binned_data (np.ndarray): Binned k-space data of shape (num_bins, extended_pe_lines, n_coils, n_freq).
      binned_count (np.ndarray): Count array of shape (num_bins, extended_pe_lines) indicating number of acquisitions.
    """
    # Reconstruct only the k-space data (without inverse FFT) from the kernel PCA outputs.
    reconstructed_kspace = (
        reconstruction.reconstruct_kspace_with_selected_components_kernel_kspace(
            kpca, X_kpca, selection, frame_shape, orig_feature_dim
        )
    )
    # reconstructed_kspace has shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq)

    # Reshape into the full k-space format expected by the binning function:
    # (total_phase_encodes, n_coils, n_freq) where total_phase_encodes = n_frames * n_phase_encodes_per_frame.
    full_reconstructed = reconstructed_kspace.reshape(
        -1, frame_shape[1], frame_shape[2]
    )

    # Call your cardiac binning function (assumed imported as cardiac_binning.bin_kspace_by_cardiac_phase).
    binned_data, binned_count = bin_kspace_by_cardiac_phase(
        r_peaks_list,
        full_reconstructed,
        num_bins=num_bins,
        n_phase_encodes_per_frame=n_phase_encodes_per_frame,
        extended_pe_lines=extended_pe_lines,
    )
    return binned_data, binned_count


def bin_kspace_by_cardiac_phase(
    r_peaks_list, img, num_bins=30, n_phase_encodes_per_frame=112, extended_pe_lines=128
):
    """
    Bin image (k-space) data by cardiac phase.

    Parameters:
      r_peaks_list : list of np.ndarray
          A list of 1D numpy arrays containing the indices of detected R-peaks for each ECG channel.
      img : np.ndarray
          The image (k-space) data as extracted from the TWIX file.
          Expected shape: (total_phase_encodes, coils, frequency_encodes).
      num_bins : int, optional
          Number of bins (frames) into which to group the data based on the cardiac cycle.
      n_phase_encodes_per_frame : int, optional
          Number of phase-encode lines per frame (e.g., 112).
      extended_pe_lines : int, optional
          The number of phase-encode lines per frame after extension (e.g., 128). Lines that do not exist
          in the original data will remain zero.

    Returns:
      binned_data : np.ndarray
          Binned k-space data of shape (num_bins, extended_pe_lines, coils, frequency_encodes).
          If multiple acquisitions fall into the same bin for the same phase-encode row, the data are averaged.

      binned_count : np.ndarray
          A 2D numpy array of shape (num_bins, extended_pe_lines) that indicates the number of acquisitions
          for each phase encode row.
    """
    # 1. Average the R-peak indices across channels.
    # Each element of r_peaks_list is assumed to be a 1D numpy array.
    avg_r_peaks = np.round(np.mean(np.vstack(r_peaks_list), axis=0)).astype(int)

    # Reshape the image data into frames.
    total_phase_encodes, n_coils, n_readout = img.shape
    if total_phase_encodes % n_phase_encodes_per_frame != 0:
        raise ValueError(
            "Total number of phase encodes is not a multiple of n_phase_encodes_per_frame."
        )
    n_frames = total_phase_encodes // n_phase_encodes_per_frame
    # Reshape so that we have (n_frames, n_phase_encodes_per_frame, coils, frequency_encodes)
    kspace_data = img.reshape(n_frames, n_phase_encodes_per_frame, n_coils, n_readout)

    # Prepare accumulation arrays for binned data.
    binned_sum = np.zeros(
        (num_bins, extended_pe_lines, n_coils, n_readout), dtype=kspace_data.dtype
    )
    binned_count = np.zeros((num_bins, extended_pe_lines), dtype=np.int64)

    # Loop over every frame and phase-encode line.
    for frame in range(n_frames):
        for row in range(n_phase_encodes_per_frame):
            # Global index corresponding to this acquisition (one-to-one with ECG sample)
            global_index = frame * n_phase_encodes_per_frame + row

            # Identify the cardiac cycle (i.e. between which R-peaks this line lies)
            cycle_idx = np.searchsorted(avg_r_peaks, global_index, side="right") - 1
            # Skip if before the first R-peak or beyond the last complete cycle.
            if cycle_idx < 0 or cycle_idx >= len(avg_r_peaks) - 1:
                continue

            cycle_start = avg_r_peaks[cycle_idx]
            cycle_end = avg_r_peaks[cycle_idx + 1]
            # Compute fraction through the current cardiac cycle.
            fraction = (global_index - cycle_start) / (cycle_end - cycle_start)
            # Map the fraction to a bin index.
            bin_idx = int(np.floor(fraction * num_bins))
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1

            # Accumulate the k-space line for the appropriate bin and phase-encode row.
            binned_sum[bin_idx, row] += kspace_data[frame, row]
            binned_count[bin_idx, row] += 1

    # Average the accumulated data when multiple lines fall into the same bin/row.
    binned_data = np.zeros_like(binned_sum)
    for b in range(num_bins):
        for r in range(extended_pe_lines):
            if binned_count[b, r] > 0:
                binned_data[b, r] = binned_sum[b, r] / binned_count[b, r]

    return binned_data, binned_count
