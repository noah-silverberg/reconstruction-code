#!/usr/bin/env python3
import twixtools
import numpy as np
import imageio
from collections import defaultdict


def homodyne_reconstruct_frame(
    mdb_list,
    full_phase=128,
    full_readout=200,
    crop_lower=8,
    crop_upper=136,
    full_target_readout=128,
    phase_offset=32,
    acquired_lines=40,
    return_kspace=False,
):
    """
    Reconstruct one frame using a basic homodyne method.

    Parameters:
      mdb_list : list of mdb blocks belonging to one frame (assumed image scans)
      full_phase : final number of phase encoding lines in k-space (e.g. 128)
      full_readout : original readout length (e.g. 200)
      crop_lower, crop_upper : indices for cropping the readout dimension (to 128)
      full_target_readout : final readout size (128)
      phase_offset : offset to place measured phase lines into full k-space
      acquired_lines : number of measured phase lines (e.g. 40)
      return_kspace : if True, also return the combined k-space magnitude

    Returns:
      img : reconstructed 2D image (normalized) using homodyne reconstruction
      [kspace] : (optional) combined k-space magnitude (root-sum-of-squares over channels)
    """
    # Determine number of channels from first mdb block
    sample_data = np.array(mdb_list[0].data)
    n_channels = sample_data.shape[0]

    # Create a full k-space array (zeros for missing data)
    kspace_full = np.zeros(
        (full_phase, full_target_readout, n_channels), dtype=np.complex64
    )

    # Insert each mdb block's cropped data into its proper k-space row.
    for mdb in mdb_list:
        data = np.array(mdb.data)  # shape: (channels, full_readout)
        data_cropped = data[:, crop_lower:crop_upper]  # shape: (channels, 128)
        measured_line = mdb.cLin  # assumed to be 0 to (acquired_lines-1)
        full_line = measured_line + phase_offset
        # If a line is already filled (e.g. if there are multiple entries for the same line), raise an error.
        if np.any(kspace_full[full_line]):
            raise ValueError(f"Duplicate data for line {full_line}")
        else:
            kspace_full[full_line] = data_cropped.T

    # Reconstruct each channel with homodyne correction
    imgs = []
    k_lowpass_channels = []
    k_weighted_channels = []
    for ch in range(n_channels):
        k_ch = kspace_full[:, :, ch]

        # Automatically calculate symmetric region lines:
        symmetric_region_lines = (acquired_lines - phase_offset) * 2
        non_sym_count = acquired_lines - symmetric_region_lines

        # Adjustable: the number of non-symmetric lines that get full 2 weighting.
        full2_start = non_sym_count // 2

        # Initialize the 1D weight vector for the acquired data.
        w = np.empty(acquired_lines, dtype=np.float32)

        # For the top (non-symmetric) region:
        # Upper half gets full 2 weighting.
        w[:full2_start] = 2.0
        # Lower half ramps linearly from 2 down to 1.
        if non_sym_count - full2_start > 0:
            ramp = np.linspace(
                2.0, 1.0, non_sym_count - full2_start + 2, endpoint=True
            )[1:-1]
            w[full2_start:non_sym_count] = ramp

        # For the symmetric region (bottom part), set weighting to 1.
        w[non_sym_count:] = 1.0

        # Replicate this 1D vector along the readout dimension.
        weight_matrix = np.tile(w[:, None], (1, full_target_readout))

        k_weighted = k_ch.copy()
        k_weighted[phase_offset : phase_offset + acquired_lines, :] *= weight_matrix
        k_weighted_channels.append(k_weighted)

        # Standard IFFT reconstruction of the weighted k-space:
        img_full = np.fft.fftshift(np.fft.ifft2(k_weighted))

        # Create a low-pass filter to extract the phase information.
        # Here we use a simple rectangular window covering the acquired k-space region.
        lowpass_filter = np.zeros_like(k_ch)
        lowpass_filter[phase_offset : phase_offset + acquired_lines, :] = 1.0

        k_lowpass = k_ch * lowpass_filter
        k_lowpass_channels.append(k_lowpass)
        img_lowres = np.fft.fftshift(np.fft.ifft2(k_lowpass))

        # Estimate the phase from the low-resolution image
        phase_est = np.angle(img_lowres)

        # Correct the full image phase by multiplying by exp(-i*phase_est)
        img_corrected = img_full * np.exp(-1j * phase_est)
        # Take the real part as the homodyne reconstructed image
        img_homodyne = np.real(img_corrected)
        imgs.append(img_homodyne)

    # Combine channels using a root-sum-of-squares
    img_combined = np.sqrt(np.sum(np.array(imgs) ** 2, axis=0))
    img_combined = img_combined / np.max(img_combined)  # normalize

    if return_kspace:
        k_lowpass_combined = np.sqrt(np.sum(np.abs(k_lowpass_channels) ** 2, axis=0))
        k_weighted_combined = np.sqrt(np.sum(np.abs(k_weighted_channels) ** 2, axis=0))

        return img_combined, k_weighted_combined, k_lowpass_combined
    else:
        return img_combined


def main():
    filename = "Cine.dat"
    twix = twixtools.read_twix(filename)
    meas = twix[-1]

    # Get image mdb blocks from the measurement
    image_mdbs = []
    for mdb in meas["mdb"]:
        try:
            if mdb.is_image_scan():
                image_mdbs.append(mdb)
        except Exception:
            continue

    # Group the image mdb blocks by frame (using mdb.cRep)
    frames_dict = defaultdict(list)
    for mdb in image_mdbs:
        frames_dict[mdb.cRep].append(mdb)

    frame_numbers = sorted(frames_dict.keys())
    print("Reconstructing frames using homodyne for frames:", frame_numbers)

    recon_frames = []
    weighted_frames = []
    lowpass_frames = []
    for rep in frame_numbers:
        mdb_list = frames_dict[rep]
        img, k_weighted, k_lowpass = homodyne_reconstruct_frame(
            mdb_list, return_kspace=True
        )
        recon_frames.append(img)
        weighted_frames.append(k_weighted)
        lowpass_frames.append(k_lowpass)

    gif_frames = [(img * 255).astype(np.uint8) for img in recon_frames]
    imageio.mimsave("homodyne_cine.gif", gif_frames, duration=0.125)
    print("Saved reconstructed cine as homodyne_cine.gif")

    # Create lowpass k-space GIF with center grid lines.
    lowpass_gif = []
    for ksp in lowpass_frames:
        # Log-scale for better dynamic range.
        disp = np.log(1 + np.abs(ksp))
        disp = (disp - disp.min()) / (disp.max() - disp.min())
        disp_uint8 = (disp * 255).astype(np.uint8)
        # Convert to RGB.
        disp_rgb = np.stack([disp_uint8] * 3, axis=-1)
        h, w, _ = disp_rgb.shape
        cx, cy = w // 2, h // 2
        # Draw vertical center line.
        disp_rgb[:, cx, :] = [255, 0, 0]
        # Draw horizontal center line.
        disp_rgb[cy, :, :] = [255, 0, 0]
        lowpass_gif.append(disp_rgb)

    imageio.mimsave("homodyne_lowpass_kspace.gif", lowpass_gif, duration=0.125)
    print("Saved filled k-space GIF as homodyne_lowpass_kspace.gif")

    # Create weighted k-space GIF with center grid lines.
    weighted_gif = []
    for ksp in weighted_frames:
        # Log-scale for better dynamic range.
        disp = np.log(1 + np.abs(ksp))
        disp = (disp - disp.min()) / (disp.max() - disp.min())
        disp_uint8 = (disp * 255).astype(np.uint8)
        # Convert to RGB.
        disp_rgb = np.stack([disp_uint8] * 3, axis=-1)
        h, w, _ = disp_rgb.shape
        cx, cy = w // 2, h // 2
        # Draw vertical center line.
        disp_rgb[:, cx, :] = [255, 0, 0]
        # Draw horizontal center line.
        disp_rgb[cy, :, :] = [255, 0, 0]
        weighted_gif.append(disp_rgb)

    imageio.mimsave("homodyne_weighted_kspace.gif", weighted_gif, duration=0.125)
    print("Saved filled k-space GIF as homodyne_weighted_kspace.gif")


if __name__ == "__main__":
    main()
