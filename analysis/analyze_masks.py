from PIL import Image
import numpy as np
import os
from collections import Counter

def analyze_masks_and_list_exceptions(masks_dir='data/masks'):
    class_grayscale_presence = {}
    class_to_grayscale_map = {}
    exceptions_list = {}

    for class_name in os.listdir(masks_dir):
        mask_class_dir = os.path.join(masks_dir, class_name)
        grayscale_presence = Counter()
        image_presence_map = {}

        # First pass: identify the most common grayscale value
        for mask_name in os.listdir(mask_class_dir):
            mask_path = os.path.join(mask_class_dir, mask_name)
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)

            unique_values = np.unique(mask_array)
            unique_values = unique_values[unique_values != 0]

            grayscale_presence.update(unique_values)
            image_presence_map[mask_name] = unique_values

        if grayscale_presence:
            most_common_value, most_common_presence = grayscale_presence.most_common(1)[0]
            class_grayscale_presence[class_name] = (most_common_value, most_common_presence)
            class_to_grayscale_map[class_name] = most_common_value

            # Second pass: check for images missing the most common grayscale value
            missing_value_filenames = [filename for filename, values in image_presence_map.items() if most_common_value not in values]
            exceptions_list[class_name] = missing_value_filenames
        else:
            class_grayscale_presence[class_name] = (None, 0)  # No nonzero value found
            class_to_grayscale_map[class_name] = None

    # Display results
    for class_name, (most_common_value, most_common_presence) in class_grayscale_presence.items():
        print(f"Class: {class_name}, Most Common Nonzero Grayscale Value (by presence): {most_common_value}, Presence Count: {most_common_presence}")
        if class_name in exceptions_list and exceptions_list[class_name]:
            print(f"    Images without the most common grayscale value ({most_common_value}): {exceptions_list[class_name]}")
        else:
            print("    All images contain the most common grayscale value.")

    return class_to_grayscale_map

