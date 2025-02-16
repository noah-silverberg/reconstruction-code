#!/usr/bin/env python3
import imageio


def crop_gif(input_filename, output_filename, crop_box):
    """
    Crop each frame of a GIF to the specified rectangle.

    Parameters:
      input_filename (str): Path to the input GIF.
      output_filename (str): Path to save the cropped GIF.
      crop_box (tuple): (x_min, y_min, x_max, y_max) defining the crop rectangle.
                        (x_min, y_min) is the top-left corner,
                        (x_max, y_max) is the bottom-right corner.
    """
    # Open the input GIF
    reader = imageio.get_reader(input_filename)
    meta = reader.get_meta_data()
    duration = meta.get("duration", 0.125)

    # Create a writer for the output GIF.
    writer = imageio.get_writer(output_filename, duration=duration)

    for frame in reader:
        # Crop frame. Frame shape assumed to be (height, width, channels).
        x_min, y_min, x_max, y_max = crop_box
        cropped = frame[y_min:y_max, x_min:x_max]
        writer.append_data(cropped)

    writer.close()
    reader.close()
    print(f"Cropped GIF saved as {output_filename}")


def main():
    input_filename = "conjugate_symmetry_cine.gif"  # Change to your input GIF filename
    output_filename = (
        "conjugate_symmetry_cine_cropped.gif"  # Change to desired output filename
    )

    # Define the crop rectangle: (x_min, y_min, x_max, y_max)
    # For example, to crop 50 pixels from the left and top, and then take a 100x100 area:
    crop_box = (24, 0, 105, 128)

    crop_gif(input_filename, output_filename, crop_box)


if __name__ == "__main__":
    main()
