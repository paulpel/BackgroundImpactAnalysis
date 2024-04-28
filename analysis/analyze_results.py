import pandas as pd
import ast
from PIL import Image
import numpy as np
import os

from analysis.analyze_masks import analyze_masks_and_list_exceptions

def categorize_data(data, column, num_categories):
    """ Categorize the data based on quantiles for a specific column """
    # Calculate quantiles including the lowest and highest bounds (0 and 1)
    quantiles = data[column].quantile(np.linspace(0, 1, num_categories + 1)).unique()  # Ensures unique quantiles
    # Generate labels, ensuring the number of labels is one fewer than the number of unique quantiles
    category_labels = [f"{int(100 * np.linspace(0, 1, len(quantiles))[i])}-{int(100 * np.linspace(0, 1, len(quantiles))[i+1])} percentile" 
                       for i in range(len(quantiles)-1)]
    # Assign categories based on bins
    data[f'{column}_category'] = pd.cut(data[column], bins=quantiles, labels=category_labels, include_lowest=True)
    return data

def load_and_process_data(file_path, base_dir, masks_dir):
    # Load the dataset
    data = pd.read_csv(file_path)
    confidence_columns = [col for col in data.columns if 'confidence' in col]
    
    for column in confidence_columns:
        data[column] = data[column].apply(lambda x: ast.literal_eval(x))
    
    # Initialize additional data columns
    data['width'] = 0
    data['height'] = 0
    data['object_percentage'] = 0.0
    data['total_pixels'] = 0

    # Get the most common grayscale value for each class from masks
    class_to_grayscale_map = analyze_masks_and_list_exceptions(masks_dir)

    for index, row in data.iterrows():
        class_id = row['id'].split('_')[0]
        picture_name = row['picture_name']
        picture_base_name = os.path.splitext(picture_name)[0]

        # Load the image
        img_path = os.path.join(base_dir, 'train', class_id, picture_name)
        with Image.open(img_path) as img:
            width, height = img.size
            data.at[index, 'width'] = width
            data.at[index, 'height'] = height
            data.at[index, 'total_pixels'] = width * height

        # Load and process the mask
        mask_name = picture_base_name + '.png'
        mask_path = os.path.join(base_dir, 'masks', class_id, mask_name)
        try:
            with Image.open(mask_path) as mask:
                mask_array = np.array(mask)
                if class_id in class_to_grayscale_map:
                    relevant_value = class_to_grayscale_map[class_id]
                    object_pixels = np.sum(mask_array == relevant_value)
                    data.at[index, 'object_percentage'] = (object_pixels / data.at[index, 'total_pixels']) * 100
        except FileNotFoundError:
            print(f"Mask not found for image: {mask_name}")

    # Categorize data based on object percentage and total pixels
    data = categorize_data(data, 'object_percentage', 4)  # e.g., quartiles
    data = categorize_data(data, 'total_pixels', 4)  # e.g., quartiles

    return data


if __name__ == "__main__":

    base_dir = 'data'
    masks_dir = 'data/masks'

    file_path = 'image_confidence_scores.csv'

    processed_data = load_and_process_data(file_path, base_dir, masks_dir)
    print(processed_data.head())  # Display the first few rows of the processed data



