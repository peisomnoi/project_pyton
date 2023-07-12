"""
Utils file of project.
"""

import os
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file ``path``."""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")

    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))

    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)

    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False

    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image.copy()

    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x

    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size

    return image

"""


def stack(batch_as_list: list):
    combined_arrays = []
    known_arrays = []
    target_tensors = []
    image_files = []
    for combined_array, known_array, target_array, image_file in batch_as_list:
        combined_arrays.append(combined_array)
        known_arrays.append(known_array)
        target_tensors.append(torch.from_numpy(target_array))
        image_files.append(image_file)

    stacked_combined_arrays = torch.stack(combined_arrays, dim=0)
    stacked_known_arrays = torch.stack(known_arrays, dim=0)
    stacked_target_tensors = torch.stack(target_tensors, dim=0)

    return stacked_combined_arrays, stacked_known_arrays, stacked_target_tensors, image_files

def stack(batch_as_list: list):
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []
    for pixelated_image, known_array, target_array, image_file in batch_as_list:
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(torch.from_numpy(target_array))
        image_files.append(image_file)

    stacked_pixelated_images = np.stack(pixelated_images, axis=0)
    stacked_known_arrays = np.stack(known_arrays, axis=0)

    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
            stacked_known_arrays), target_arrays, image_files
"""


def stack(batch_as_list: list):
        # Expected list elements are 4-tuples:
        # (pixelated_image, known_array, target_array, image_file)
        n = len(batch_as_list)
        pixelated_images_dtype = batch_as_list[0][0].dtype
        known_arrays_dtype = batch_as_list[0][1].dtype
        shapes = []
        pixelated_images = []
        known_arrays = []
        target_arrays = []
        image_files = []

        for pixelated_image, known_array, target_array, image_file in batch_as_list:
            shapes.append(pixelated_image.shape)
            pixelated_images.append(pixelated_image)
            known_arrays.append(known_array)
            target_arrays.append(torch.from_numpy(target_array))
            image_files.append(image_file)

        max_shape = np.max(np.stack(shapes, axis=0), axis=0)
        stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
        stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)

        for i in range(n):
            channels, height, width = pixelated_images[i].shape
            stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
            stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]

        return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
            stacked_known_arrays), target_arrays, image_files




