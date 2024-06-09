import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict


def find_contrast_colors(colors):
    """
    Finds high and low contrast colors given a set of dominant colors.

    Args:
        colors (list): A list of RGB colors.

    Returns:
        dict: A dictionary with low and high contrast colors.
    """
    colors = np.array(colors) / 255.0
    mean_color = np.mean(colors, axis=0)
    distances = np.sqrt(np.sum((colors - mean_color) ** 2, axis=1))
    high_contrast_color = colors[np.argmax(distances)]
    low_contrast_color = mean_color

    high_contrast_color = tuple(int(x) for x in (high_contrast_color * 255))
    low_contrast_color = tuple(int(x) for x in (low_contrast_color * 255))

    return {"low": low_contrast_color, "high": high_contrast_color}


def analyze_colors(directory):
    """
    Analyze the colors in the images within a directory to find dominant colors.

    Args:
        directory (str): The path to the directory containing the images.

    Returns:
        dict: A dictionary with low and high contrast colors.
    """
    color_samples = []
    for filename in os.listdir(directory):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        data = np.array(image)

        mask = (data[:, :, 0] != 0) & (data[:, :, 1] != 0) & (data[:, :, 2] != 0)
        filtered_data = data[mask]

        color_samples.extend(filtered_data)

    if len(color_samples) == 0:
        return {"low": (0, 0, 0), "high": (255, 255, 255)}

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(color_samples)
    dominant_colors = kmeans.cluster_centers_

    return find_contrast_colors(dominant_colors)


if __name__ == "__main__":
    base_dir = "data/modified/no_bg"
    classes = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    color_dict = defaultdict(dict)

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        color_dict[class_name] = analyze_colors(class_dir)

    with open("class_colors.json", "w") as file:
        json.dump(color_dict, file, indent=4)

    print("Color analysis data saved to 'class_colors.json'.")
