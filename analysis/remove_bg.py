from PIL import Image
import numpy as np
import os

from analysis.analyze_masks import analyze_masks_and_list_exceptions

def remove_background_and_save(input_dir, mask_dir, output_dir, grayscale_values):
    for class_name, value in grayscale_values.items():
        current_input_dir = os.path.join(input_dir, class_name)
        current_mask_dir = os.path.join(mask_dir, class_name)
        current_output_dir = os.path.join(output_dir, class_name)

        print(current_input_dir, current_mask_dir, current_output_dir)
        print(os.path.exists(current_input_dir), os.path.exists(current_mask_dir), os.path.exists(current_output_dir))

        # Create the output directory if it doesn't exist
        os.makedirs(current_output_dir, exist_ok=True)

        for mask_name in os.listdir(current_mask_dir):
            base_filename, _ = os.path.splitext(mask_name)
            image_filename = f"{base_filename}.JPEG"  # Assuming original images are .jpg
            image_path = os.path.join(current_input_dir, image_filename)
            mask_path = os.path.join(current_mask_dir, mask_name)

            print(image_path, mask_path, os.getcwd())
            print(os.path.exists(image_path), os.path.exists(mask_path))

            if not os.path.exists(image_path):
                print(f"No original image found for {mask_name} in {class_name}, skipping.")
                continue

            # Open the original image and mask
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            # Convert mask to a binary mask for the specific grayscale value
            object_mask = np.array(mask) == value

            # Prepare a black background of the same size as the original image
            background = Image.new("RGB", image.size)

            # Apply the mask: for pixels where the mask matches, copy the original image's pixels
            result_image = np.where(object_mask[:, :, None], np.array(image), np.array(background))

            # Save the processed image
            Image.fromarray(result_image.astype(np.uint8)).save(os.path.join(current_output_dir, f"{base_filename}.png"))

        print(f"Processed class {class_name}")

# Example usage
input_directory = 'data/train'
mask_directory = 'data/masks'
output_directory = 'data/modified/no_bg'

# Example dictionary, replace it with your actual grayscale values
grayscale_values = analyze_masks_and_list_exceptions()

remove_background_and_save(input_directory, mask_directory, output_directory, grayscale_values)

