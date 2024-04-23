from PIL import Image
import numpy as np
import os

from analysis.analyze_masks import analyze_masks_and_list_exceptions

def apply_scenic_backgrounds(input_dir, mask_dir, output_dir, background_dir, backgrounds_info, grayscale_values):
    """Applies different scenic backgrounds to images using masks, skipping images that are already processed."""
    for class_name, value in grayscale_values.items():
        current_input_dir = os.path.join(input_dir, class_name)
        current_mask_dir = os.path.join(mask_dir, class_name)

        for scenario, background_file in backgrounds_info.items():
            scenario_output_dir = os.path.join(output_dir, scenario, class_name)
            os.makedirs(scenario_output_dir, exist_ok=True)

        for mask_name in os.listdir(current_mask_dir):
            base_filename, _ = os.path.splitext(mask_name)
            image_filename = f"{base_filename}.JPEG"
            image_path = os.path.join(current_input_dir, image_filename)
            mask_path = os.path.join(current_mask_dir, mask_name)

            if not os.path.exists(image_path):
                print(f"No original image found for {mask_name} in {class_name}, skipping.")
                continue

            image = Image.open(image_path).convert("RGBA")
            mask = Image.open(mask_path).convert("L") 

            object_mask = np.array(mask) == value
            object_mask = np.expand_dims(object_mask, axis=2)  # Expand dims to apply on image

            for scenario, background_file in backgrounds_info.items():
                scenario_output_dir = os.path.join(output_dir, scenario, class_name)
                output_image_path = os.path.join(scenario_output_dir, f"{base_filename}.png")

                # Check if the output image already exists, skip if it does
                if os.path.exists(output_image_path):
                    print(f"Skipping already processed {output_image_path}")
                    continue

                background_path = os.path.join(background_dir, background_file)
                background_image = Image.open(background_path).convert("RGBA")
                background_resized = background_image.resize(image.size, Image.Resampling.LANCZOS)

                result_image = np.where(object_mask, np.array(image), np.array(background_resized))
                Image.fromarray(result_image.astype(np.uint8)).save(output_image_path)

                print(f"Processed and saved {output_image_path}")

            print(f"Completed processing all scenarios for {image_filename} in class {class_name}.")

if __name__ == "__main__":
    input_directory = 'data/train'
    mask_directory = 'data/masks'
    output_directory = 'data/modified'
    background_directory = 'data/scenarios'

    backgrounds_info = {
        "city": "city.jpg",
        "jungle": "jungle.jpg",
        "desert": "desert.jpg",
        "water": "water.jpg",
        "sky": "sky.jpg",
        "indoor": "indoor.jpg",
        "mountain": "mountain.jpg",
        "snow": "snow.jpg"
    }

    grayscale_values = analyze_masks_and_list_exceptions()

    apply_scenic_backgrounds(input_directory, mask_directory, output_directory, background_directory, backgrounds_info, grayscale_values)

