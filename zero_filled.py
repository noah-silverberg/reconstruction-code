#!/usr/bin/env python3
import twixtools
import numpy as np
import imageio
import matplotlib.pyplot as plt
from collections import defaultdict


def reconstruct_frame(
    mdb_list,
    full_phase=128,
    full_readout=200,
    crop_lower=64,
    crop_upper=192,
    full_target_readout=128,
    phase_offset=12,
):
    """
    Reconstruct one frame from a list of mdb blocks.

    Parameters:
      mdb_list : list of mdb blocks belonging to one frame (assumed image scans)
      full_phase : full number of phase encoding lines in the final k-space (from protocol, e.g. 128)
      full_readout : readout length in raw data (200)
      crop_lower, crop_upper : indices to crop raw readout data to remove oversampling (yielding 128 samples)
      full_target_readout : final readout size after cropping (128)
      phase_offset : offset to embed measured phase lines into full k-space (e.g. measured index + 12)

    Returns:
      img : reconstructed 2D image (magnitude, normalized)
    """
    # We know each mdb.data is (channels, full_readout)
    # Determine number of channels from first mdb block
    sample_data = np.array(mdb_list[0].data)
    n_channels = sample_data.shape[0]

    # Create a full k-space array of shape (full_phase, full_target_readout, channels)
    kspace_full = np.zeros(
        (full_phase, full_target_readout, n_channels), dtype=np.complex64
    )

    # For each mdb block, embed its cropped data into the appropriate row.
    # Here we assume mdb.cLin gives the measured line index (0..39).
    for mdb in mdb_list:
        # Convert data to numpy array
        data = np.array(mdb.data)  # shape: (channels, full_readout)
        # Crop the readout dimension: keep columns crop_lower:crop_upper -> shape (channels, 128)
        data_cropped = data[:, crop_lower:crop_upper]
        # Determine the measured phase encoding index.
        # We assume mdb.cLin is an integer from 0 to (number of acquired lines-1)
        measured_line = mdb.cLin
        # Map the measured line into the full k-space.
        full_line = measured_line + phase_offset  # e.g. measured 0->12, measured 39->51
        # If a line is already filled (e.g. if there are multiple entries for the same line), raise an error.
        if np.any(kspace_full[full_line]):
            raise ValueError(f"Duplicate data for line {full_line}")
        else:
            kspace_full[full_line] = data_cropped.T

    # Now, for each channel, perform 2D inverse FFT to reconstruct the image.
    # We assume that the phase encoding dimension is the first axis and frequency encoding is second.
    imgs = []
    for ch in range(n_channels):
        # Get k-space for this channel.
        k_ch = kspace_full[:, :, ch]
        # 2D inverse FFT
        img_ch = np.fft.ifft2(k_ch)
        # fftshift the result to get the image in the right order.
        img_ch = np.fft.fftshift(img_ch)
        imgs.append(img_ch)
    # Combine channels via root-sum-of-squares
    img_combined = np.sqrt(np.sum(np.abs(np.array(imgs)) ** 2, axis=0))

    # Normalize the image for display
    img_combined = img_combined / np.max(img_combined)
    return img_combined


def main():
    filename = "Cine.dat"
    # Read the twix data
    twix = twixtools.read_twix(filename)
    meas = twix[-1]

    # Get the list of mdb blocks that are image scans
    image_mdbs = []
    for mdb in meas["mdb"]:
        try:
            if mdb.is_image_scan():
                image_mdbs.append(mdb)
        except Exception:
            continue

    # Group the image mdb blocks by frame number (mdb.cRep)
    frames_dict = defaultdict(list)
    for mdb in image_mdbs:
        frames_dict[mdb.cRep].append(mdb)

    # Sort frames by frame number
    frame_numbers = sorted(frames_dict.keys())
    print("Reconstructing frames for frame numbers:", frame_numbers)

    # List to hold reconstructed image frames
    recon_frames = []

    # Loop over each frame and reconstruct the image
    for rep in frame_numbers:
        mdb_list = frames_dict[rep]
        img = reconstruct_frame(mdb_list)
        recon_frames.append(img)

    # Create a GIF from the reconstructed frames.
    # First, convert each frame to an 8-bit grayscale image.
    gif_frames = []
    for img in recon_frames:
        # Scale to 0-255 and convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        # Optionally, use a colormap for visualization; here we keep grayscale.
        gif_frames.append(img_uint8)

    output_gif = "cine_recon.gif"
    # Save GIF using imageio (duration between frames in seconds)
    imageio.mimsave(output_gif, gif_frames, duration=0.125)
    print(f"Saved reconstructed cine as {output_gif}")


if __name__ == "__main__":
    main()
