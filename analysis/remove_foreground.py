from PIL import Image
import numpy as np
import os

from analysis.analyze_masks import analyze_masks_and_list_exceptions


def save_background_only(input_dir, mask_dir, output_dir, grayscale_values):
    """
    Save images with only the background, removing the foreground based on the specified grayscale values from masks.

    Args:
        input_dir (str): Directory containing the original images.
        mask_dir (str): Directory containing the mask images.
        output_dir (str): Directory to save the images with only the background.
        grayscale_values (dict): Dictionary mapping class names to grayscale values to be used for masking.
    """
    for class_name, value in grayscale_values.items():
        current_input_dir = os.path.join(input_dir, class_name)
        current_mask_dir = os.path.join(mask_dir, class_name)
        current_output_dir = os.path.join(output_dir, class_name)

        os.makedirs(current_output_dir, exist_ok=True)

        for mask_name in os.listdir(current_mask_dir):
            base_filename, _ = os.path.splitext(mask_name)
            output_image_path = os.path.join(current_output_dir, f"{base_filename}.png")

            if os.path.exists(output_image_path):
                print(f"Skipping already processed {output_image_path}")
                continue

            image_filename = f"{base_filename}.JPEG"
            image_path = os.path.join(current_input_dir, image_filename)
            mask_path = os.path.join(current_mask_dir, mask_name)

            if not os.path.exists(image_path):
                print(
                    f"No original image found for {mask_name} in {class_name}, skipping."
                )
                continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            object_mask = np.array(mask) != value

            result_image = np.where(object_mask[:, :, None], np.array(image), 0)

            Image.fromarray(result_image.astype(np.uint8)).save(output_image_path)

        print(f"Processed class {class_name}")


if __name__ == "__main__":
    input_directory = "data/train"
    mask_directory = "data/masks"
    output_directory = "data/modified/no_foreground"

    grayscale_values = analyze_masks_and_list_exceptions()

    save_background_only(
        input_directory, mask_directory, output_directory, grayscale_values
    )
