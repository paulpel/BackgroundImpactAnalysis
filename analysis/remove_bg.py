from PIL import Image
import numpy as np
import os

from analysis.analyze_masks import analyze_masks_and_list_exceptions


def remove_background_and_save(input_dir, mask_dir, output_dir, grayscale_values):
    """
    Remove background from images using the specified grayscale values from masks and save the results.

    Args:
        input_dir (str): Directory containing the original images.
        mask_dir (str): Directory containing the mask images.
        output_dir (str): Directory to save the images with the background removed.
        grayscale_values (dict): Dictionary mapping class names to grayscale values to be used for masking.
    """
    for class_name, value in grayscale_values.items():
        current_input_dir = os.path.join(input_dir, class_name)
        current_mask_dir = os.path.join(mask_dir, class_name)
        current_output_dir = os.path.join(output_dir, class_name)

        print(current_input_dir, current_mask_dir, current_output_dir)
        print(
            os.path.exists(current_input_dir),
            os.path.exists(current_mask_dir),
            os.path.exists(current_output_dir),
        )
        os.makedirs(current_output_dir, exist_ok=True)

        for mask_name in os.listdir(current_mask_dir):
            base_filename, _ = os.path.splitext(mask_name)
            image_filename = f"{base_filename}.JPEG"
            image_path = os.path.join(current_input_dir, image_filename)
            mask_path = os.path.join(current_mask_dir, mask_name)

            print(image_path, mask_path, os.getcwd())
            print(os.path.exists(image_path), os.path.exists(mask_path))

            if not os.path.exists(image_path):
                print(
                    f"No original image found for {mask_name} in {class_name}, skipping."
                )
                continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            object_mask = np.array(mask) == value

            background = Image.new("RGB", image.size)

            result_image = np.where(
                object_mask[:, :, None], np.array(image), np.array(background)
            )

            Image.fromarray(result_image.astype(np.uint8)).save(
                os.path.join(current_output_dir, f"{base_filename}.png")
            )

        print(f"Processed class {class_name}")


if __name__ == "__main__":
    input_directory = "data/train"
    mask_directory = "data/masks"
    output_directory = "data/modified/no_bg"

    grayscale_values = analyze_masks_and_list_exceptions()

    remove_background_and_save(
        input_directory, mask_directory, output_directory, grayscale_values
    )
