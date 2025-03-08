import numpy as np


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


def homodyne_binned_data(binned_data, binned_count):
    """
    Apply homodyne correction to binned k-space data without enforcing continuity,
    and return the reconstructed images.

    Instead of enforcing that the acquired region is contiguous, this function
    assigns a weight of 1 to every acquired phase-encode row that has its symmetric
    counterpart also measured, and a weight of 2 to rows without a measured counterpart.
    For each bin, the function:
      1. Applies the weight to the acquired rows.
      2. Reconstructs a low-resolution image from the acquired data (using a lowpass filter)
         to estimate the phase.
      3. Corrects the phase of the full weighted image.
      4. Reconstructs the corrected image for each coil.
      5. Combines the coil images via root-sum-of-squares and returns the final real image.

    Parameters:
      binned_data : np.ndarray
          Complex k-space data of shape
          (num_bins, extended_pe_lines, n_coils, frequency_encodes).
      binned_count : np.ndarray
          2D array of shape (num_bins, extended_pe_lines) indicating, for each phase-encode
          row in each bin, how many acquisitions contributed.

    Returns:
      recon_images : np.ndarray
          Reconstructed images of shape (num_bins, image_height, image_width) obtained by
          taking the real part of the homodyne-corrected images and combining channels via
          root-sum-of-squares.
    """
    num_bins, n_phase, n_coils, n_readout = binned_data.shape
    recon_images = []

    # Define the center of k-space (phase dimension) to compute symmetry.
    center = n_phase / 2.0  # may be fractional

    for b in range(num_bins):
        # Determine the weight for each phase-encode row in this bin.
        # For each acquired row (binned_count > 0):
        #   - If its symmetric counterpart is also acquired, weight = 1.
        #   - Otherwise, weight = 2.
        w = np.zeros(n_phase, dtype=np.float32)
        for r in range(n_phase):
            if binned_count[b, r] > 0:
                # Compute symmetric index: r_sym = round(2*center - r)
                r_sym = int(round(2 * center - r))
                if 0 <= r_sym < n_phase and binned_count[b, r_sym] > 0:
                    w[r] = 1.0
                else:
                    w[r] = 2.0
        # Create weight matrix along the readout dimension.
        weight_matrix = np.tile(w[:, None], (1, n_readout))

        # Process each coil for the current bin.
        coil_imgs = []
        for ch in range(n_coils):
            k_ch = binned_data[b, :, ch, :]  # shape: (n_phase, n_readout)

            # Apply the weight to the k-space data.
            k_weighted = k_ch * weight_matrix

            # Construct a simple lowpass filter that selects acquired rows.
            lowpass_filter = (binned_count[b, :] > 0).astype(np.float32)
            lowpass_matrix = np.tile(lowpass_filter[:, None], (1, n_readout))
            k_lowpass = k_ch * lowpass_matrix

            # Reconstruct images from weighted and lowpass data.
            img_full = np.fft.fftshift(np.fft.ifft2(k_weighted))
            img_lowres = np.fft.fftshift(np.fft.ifft2(k_lowpass))

            # Estimate the phase from the low-resolution image.
            phase_est = np.angle(img_lowres)

            # Correct the full image phase.
            img_corrected = img_full * np.exp(-1j * phase_est)

            # Take the real part as the final corrected image for this coil.
            img_real = np.real(img_corrected)
            coil_imgs.append(img_real)

        # Combine the coil images using root-sum-of-squares.
        coil_imgs = np.array(coil_imgs)  # shape: (n_coils, image_height, image_width)
        sos_img = np.sqrt(np.sum(coil_imgs**2, axis=0))
        recon_images.append(sos_img)

    return np.array(recon_images)


def reconstruct_image_from_binned_kspace(binned_data):
    """
    Reconstruct images from binned k-space data using basic sum-of-squares (SoS) across coils.

    Parameters:
      binned_data : np.ndarray
          Binned k-space data of shape (num_bins, extended_pe_lines, coils, frequency_encodes).

    Returns:
      recon_images : np.ndarray
          Reconstructed images of shape (num_bins, image_height, image_width).
          The reconstruction is performed by applying an inverse 2D FFT to each coil's data
          and then combining the coil images with a sum-of-squares operation.
    """
    num_bins, n_pe, n_coils, n_readout = binned_data.shape
    recon_images = []

    for b in range(num_bins):
        coil_images = []
        for c in range(n_coils):
            # Optionally apply an FFT shift if needed.
            # kspace = np.fft.ifftshift(binned_data[b, :, c, :])
            kspace = binned_data[b, :, c, :]
            # Inverse 2D FFT to reconstruct the coil image.
            image = np.fft.ifft2(kspace)
            image = np.fft.fftshift(image)
            coil_images.append(np.abs(image))
        coil_images = np.array(coil_images)
        # Combine coil images with the sum-of-squares method.
        sos_image = np.sqrt(np.sum(coil_images**2, axis=0))
        recon_images.append(sos_image)

    return np.array(recon_images)
