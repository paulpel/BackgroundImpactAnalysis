import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict


def find_contrast_colors(colors):
    """Finds high and low contrast colors given a set of dominant colors."""
    colors = np.array(colors) / 255.0  # Normalize colors to 0-1 scale
    mean_color = np.mean(colors, axis=0)  # Calculate the mean color (for low contrast)
    distances = np.sqrt(
        np.sum((colors - mean_color) ** 2, axis=1)
    )  # Euclidean distances
    high_contrast_color = colors[np.argmax(distances)]  # Color furthest from the mean
    low_contrast_color = mean_color  # Mean color

    # Ensure types are native Python integers for JSON serialization
    high_contrast_color = tuple(int(x) for x in (high_contrast_color * 255))
    low_contrast_color = tuple(int(x) for x in (low_contrast_color * 255))

    return {"low": low_contrast_color, "high": high_contrast_color}


def analyze_colors(directory):
    color_samples = []
    for filename in os.listdir(directory):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        data = np.array(image)

        mask = (
            (data[:, :, 0] != 0) & (data[:, :, 1] != 0) & (data[:, :, 2] != 0)
        )  # Mask to ignore black
        filtered_data = data[mask]

        color_samples.extend(filtered_data)

    if len(color_samples) == 0:
        return {"low": (0, 0, 0), "high": (255, 255, 255)}  # Default if no colors found

    kmeans = KMeans(n_clusters=5)  # Cluster colors
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

    # Save the dictionary as a JSON file
    with open("class_colors.json", "w") as file:
        json.dump(color_dict, file, indent=4)

    print("Color analysis data saved to 'class_colors.json'.")
