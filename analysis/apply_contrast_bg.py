import os
import json
import numpy as np
from PIL import Image

from analysis.analyze_masks import analyze_masks_and_list_exceptions


def apply_contrast_background(
    input_dir, mask_dir, output_base_dir, contrast_colors, grayscale_values
):
    """
    Applies contrast backgrounds to images using masks and saves them.

    Args:
        input_dir (str): Directory containing the original images without background.
        mask_dir (str): Directory containing the mask images.
        output_base_dir (str): Base directory to save the processed images with contrast backgrounds.
        contrast_colors (dict): Dictionary mapping class names to their low and high contrast colors.
        grayscale_values (dict): Dictionary mapping class names to their most common nonzero grayscale values in masks.
    """
    for class_name, value in grayscale_values.items():
        low_contrast_dir = os.path.join(output_base_dir, "low_contrast", class_name)
        high_contrast_dir = os.path.join(output_base_dir, "high_contrast", class_name)

        os.makedirs(low_contrast_dir, exist_ok=True)
        os.makedirs(high_contrast_dir, exist_ok=True)

        input_class_dir = os.path.join(input_dir, class_name)
        mask_class_dir = os.path.join(mask_dir, class_name)

        for image_file in os.listdir(input_class_dir):
            image_path = os.path.join(input_class_dir, image_file)
            mask_path = os.path.join(mask_class_dir, image_file)

            if not os.path.exists(mask_path):
                print(f"No mask found for {image_file} in {class_name}, skipping.")
                continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            object_mask = np.array(mask) == value
            object_mask = np.expand_dims(object_mask, axis=2)

            data = np.array(image)

            low_background = np.full(
                data.shape, contrast_colors[class_name]["low"], dtype=np.uint8
            )
            low_contrast_image = np.where(object_mask, data, low_background)
            Image.fromarray(low_contrast_image).save(
                os.path.join(low_contrast_dir, image_file)
            )

            high_background = np.full(
                data.shape, contrast_colors[class_name]["high"], dtype=np.uint8
            )
            high_contrast_image = np.where(object_mask, data, high_background)
            Image.fromarray(high_contrast_image).save(
                os.path.join(high_contrast_dir, image_file)
            )

        print(
            f"Processed images for class {class_name} with low and high contrast backgrounds."
        )


if __name__ == "__main__":
    input_directory = "data/modified/no_bg"
    mask_directory = "data/masks"
    output_directory = "data/modified"
    color_config_path = "class_colors.json"

    with open(color_config_path, "r") as file:
        contrast_colors = json.load(file)

    grayscale_values = analyze_masks_and_list_exceptions()

    apply_contrast_background(
        input_directory,
        mask_directory,
        output_directory,
        contrast_colors,
        grayscale_values,
    )
