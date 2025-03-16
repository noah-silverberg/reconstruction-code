import numpy as np


def reconstruct_with_selected_components_kspace(
    U, S, Vt, X_mean, selection, frame_shape
):
    """
    Reconstruct the k-space data using only the specified principal components,
    then convert it to the image domain using a 2D inverse FFT per coil, and combine
    the coils using sum-of-squares.

    Parameters:
      U, S, Vt, X_mean : SVD results from perform_pca_kspace.
      selection        : Either an integer or a list of PC indices (0-indexed) to use.
      frame_shape      : Tuple (phase_encodes, coils, frequency_encodes).

    Returns:
      images_sos       : Reconstructed images (n_bins, phase_encodes, frequency_encodes).
    """
    # Ensure selection is a list.
    if isinstance(selection, int):
        selection = [selection]

    n_frames = U.shape[0]
    # Initialize reconstruction in vector form.
    X_recon = np.zeros((n_frames, Vt.shape[1]), dtype=complex)
    for idx in selection:
        X_recon += np.outer(U[:, idx] * S[idx], Vt[idx, :])
    X_recon += X_mean

    # Reshape back to (n_frames, phase_encodes, coils, frequency_encodes).
    kspace_recon = X_recon.reshape(
        n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
    )

    # Inverse FFT per coil.
    images_recon = np.zeros(
        (n_frames, frame_shape[0], frame_shape[1], frame_shape[2]), dtype=complex
    )
    for b in range(n_frames):
        for c in range(frame_shape[1]):
            images_recon[b, :, c, :] = np.fft.fftshift(
                np.fft.ifft2(kspace_recon[b, :, c, :])
            )

    # Combine coils via sum-of-squares.
    images_sos = np.sqrt(np.sum(np.abs(images_recon) ** 2, axis=2))
    return images_sos


def reconstruct_kspace_with_selected_components_kernel_kspace(
    kpca, X_kpca, selection, frame_shape, orig_feature_dim
):
    """
    Reconstruct the k-space data using only the specified kernel PCA components,
    without converting it to the image domain (i.e. without applying inverse FFT).
    The inverse transform is applied on the real-valued representation, then split
    back into its real and imaginary parts to form complex k-space data.

    Parameters:
      kpca (KernelPCA object): Fitted KernelPCA model with fit_inverse_transform=True.
      X_kpca (np.ndarray): Kernel PCA transformed data of shape (n_frames, n_components_total).
      selection (int or list): Either an integer or a list of component indices (0-indexed) to use.
      frame_shape (tuple): Shape of a single frame in the original domain, e.g.
                           (n_phase_encodes_per_frame, n_coils, n_freq).
      orig_feature_dim (int): Original number of features per frame (before converting complex to real).

    Returns:
      reconstructed_kspace (np.ndarray): Reconstructed k-space data with shape
           (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
    """
    if isinstance(selection, int):
        selection = [selection]

    # Zero out all non-selected components.
    X_kpca_mod = np.zeros_like(X_kpca)
    X_kpca_mod[:, selection] = X_kpca[:, selection]

    # Inverse transform returns a real array of shape (n_frames, 2 * orig_feature_dim)
    X_recon_real = kpca.inverse_transform(X_kpca_mod)

    # Split back into real and imaginary parts to form complex data.
    X_recon_complex = (
        X_recon_real[:, :orig_feature_dim] + 1j * X_recon_real[:, orig_feature_dim:]
    )
    n_frames = X_recon_complex.shape[0]
    reconstructed_kspace = X_recon_complex.reshape(
        n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
    )
    return reconstructed_kspace


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


def reconstruct_frames_kspace(U, S, Vt, X_mean, n_components, frame_shape):
    """
    Reconstruct the k-space data using the first n_components principal components,
    then convert it to the image domain using a 2D inverse FFT per coil, and combine
    the coils using sum-of-squares.

    Parameters:
      U, S, Vt, X_mean : SVD results from perform_pca_kspace.
      n_components     : Number of components to use for reconstruction.
      frame_shape      : Tuple (phase_encodes, frequency_encodes).
      n_bins           : Number of bins (frames).
      n_coils          : Number of coils.

    Returns:
      images_sos     : Reconstructed images (n_bins, phase_encodes, frequency_encodes).
    """
    # Reconstruct in the k-space domain.
    X_recon = (
        U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components, :] + X_mean
    )
    # Reshape back to (n_bins, phase_encodes, coils, frequency_encodes).
    n_frames = X_recon.shape[0]
    kspace_recon = X_recon.reshape(
        n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
    )
    # Inverse FFT per coil.
    images_recon = np.zeros(
        (n_frames, frame_shape[0], frame_shape[1], frame_shape[2]), dtype=complex
    )
    for b in range(n_frames):
        for c in range(frame_shape[1]):
            images_recon[b, :, c, :] = np.fft.fftshift(
                np.fft.ifft2(kspace_recon[b, :, c, :])
            )
    # Combine coils via sum-of-squares.
    images_sos = np.sqrt(np.sum(np.abs(images_recon) ** 2, axis=2))
    return images_sos


def reconstruct_frames_kernel_kspace(kpca, X_kpca, frame_shape, orig_feature_dim):
    """
    Reconstruct the k-space data using kernel PCA's inverse transform,
    then convert it to the image domain using a 2D inverse FFT per coil,
    and combine the coils using sum-of-squares.

    Parameters:
      kpca (KernelPCA object): Fitted KernelPCA model with fit_inverse_transform=True.
      X_kpca (np.ndarray): Kernel PCA transformed data of shape (n_frames, n_components).
      frame_shape (tuple): Shape of a single frame in the original domain (n_phase_encodes_per_frame, n_coils, n_freq).
      orig_feature_dim (int): Original number of features per frame before real conversion.

    Returns:
      images_sos (np.ndarray): Reconstructed images (n_frames, n_phase_encodes_per_frame, n_freq).
    """
    # Inverse transform returns a real array of shape (n_frames, 2*orig_feature_dim)
    X_recon_real = kpca.inverse_transform(X_kpca)
    # Split back into real and imaginary parts:
    n_features = orig_feature_dim  # Original complex feature dimension.
    X_recon_complex = X_recon_real[:, :n_features] + 1j * X_recon_real[:, n_features:]
    n_frames = X_recon_complex.shape[0]
    kspace_recon = X_recon_complex.reshape(
        n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
    )

    # Inverse FFT per coil.
    images_recon = np.zeros(
        (n_frames, frame_shape[0], frame_shape[1], frame_shape[2]), dtype=complex
    )
    for b in range(n_frames):
        for c in range(frame_shape[1]):
            images_recon[b, :, c, :] = np.fft.fftshift(
                np.fft.ifft2(kspace_recon[b, :, c, :])
            )

    # Combine coils via sum-of-squares.
    images_sos = np.sqrt(np.sum(np.abs(images_recon) ** 2, axis=2))
    return images_sos


def reconstruct_with_selected_components_kernel_kspace(
    kpca, X_kpca, selection, frame_shape, orig_feature_dim, extended_pe_lines, offset
):
    """
    Reconstruct the k-space data using only the specified kernel PCA components,
    then convert it to the image domain using a 2D inverse FFT per coil, and combine
    the coils using sum-of-squares.

    Parameters:
      kpca (KernelPCA object): Fitted KernelPCA model with fit_inverse_transform=True.
      X_kpca (np.ndarray): Kernel PCA transformed data of shape (n_frames, n_components_total).
      selection (int or list): Either an integer or a list of component indices (0-indexed) to use.
      frame_shape (tuple): Shape of a single frame in the original domain (n_phase_encodes_per_frame, n_coils, n_freq).
      orig_feature_dim (int): Original number of features per frame before real conversion.
      extended_pe_lines (int): Number of phase encode lines after extension.
      offset : offset to place measured phase lines


    Returns:
      images_sos (np.ndarray): Reconstructed images with shape (n_frames, n_phase_encodes_per_frame, n_freq).
    """
    # Ensure selection is a list.
    if isinstance(selection, int):
        selection = [selection]

    X_kpca_mod = np.zeros_like(X_kpca)
    X_kpca_mod[:, selection] = X_kpca[:, selection]

    X_recon_real = kpca.inverse_transform(X_kpca_mod)
    n_features = orig_feature_dim
    X_recon_complex = X_recon_real[:, :n_features] + 1j * X_recon_real[:, n_features:]
    n_frames = X_recon_complex.shape[0]
    kspace_recon = X_recon_complex.reshape(
        n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
    )
    kspace_zerofilled = np.zeros(
        (n_frames, extended_pe_lines, frame_shape[1], frame_shape[2]), dtype=complex
    )
    kspace_zerofilled[:, offset : offset + frame_shape[0], :, :] = kspace_recon

    images_recon = np.zeros(
        (n_frames, extended_pe_lines, frame_shape[1], frame_shape[2]), dtype=complex
    )
    for b in range(n_frames):
        for c in range(frame_shape[1]):
            images_recon[b, :, c, :] = np.fft.fftshift(
                np.fft.ifft2(kspace_zerofilled[b, :, c, :])
            )

    images_sos = np.sqrt(np.sum(np.abs(images_recon) ** 2, axis=2))
    return images_sos
