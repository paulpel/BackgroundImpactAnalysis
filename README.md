## Project Description

### Overview

This project is focused on automating the process of analyzing and modifying image datasets using deep learning models. It encompasses a range of scripts for generating image masks, removing and replacing backgrounds, applying scenic backgrounds, and comparing the effects of these modifications on image classification. The goal is to streamline the preparation and analysis of image datasets for various applications, such as training machine learning models, performing image segmentation, and studying the impact of different backgrounds on classification accuracy.

### Features

- **Image Mask Generation and Analysis**: Automatically generate image masks and analyze them to identify common grayscale values and detect inconsistencies.
- **Background Removal**: Remove backgrounds from images based on the generated masks.
- **Contrast and Scenic Background Application**: Apply contrast or scenic backgrounds to images, creating diverse datasets for analysis.
- **Image Classification and Comparison**: Use pre-trained models like ResNet and ConvNeXt to classify original and modified images, comparing the top-5 class predictions and confidence scores.
- **Comprehensive Reporting**: Save the classification results and analysis in easily interpretable formats such as CSV and JSON files.

### Scripts

1. **generate_masks_and_overlays.py**: Generates masks and overlays for images using a pre-trained DeepLabV3 model.
2. **analysis/seg_model_1.py**: Supports mask generation by loading the model and processing images.
3. **analysis/mask_analysis.py**: Analyzes mask images to find the most common grayscale values and list exceptions.
4. **remove_background.py**: Removes the background from images using the specified grayscale values from masks.
5. **color_analysis.py**: Analyzes the colors in images to determine high and low contrast colors for each class.
6. **apply_contrast_background.py**: Applies contrast backgrounds to images using masks and pre-defined color configurations.
7. **apply_scenic_backgrounds.py**: Applies various scenic backgrounds to images using masks.
8. **compare_images_convnext.py**: Compares original and modified images using a pre-trained ConvNeXt model and saves the classification results.

### Dependencies

- **Python**: Core programming language used for script development.
- **PyTorch**: Used for deep learning models and image transformations.
- **Torchvision**: Provides pre-trained models and additional utilities for image processing.
- **PIL**: Used for image loading, manipulation, and saving.
- **NumPy**: Used for array operations and numerical computations.
- **Pandas**: Used for saving results to CSV files.
- **OS**: Used for file and directory operations.
- **Scikit-learn**: Used for clustering in color analysis.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [Main Script](#main-script)
  - [Segmentation Script](#segmentation-script)
  - [Mask Script](#mask-script)
  - [BackgroundRemoval Script](#backgroundremoval-script)
  - [Color Script](#color-script)
  - [Foreground Script](#foreground-script)
  - [Contrast Script](#contrast-script)
  - [Scenic Script](#scenic-script)
  - [Resnet Script](#resnet-script)
  - [Convnext Script](#convnext-script)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/paulpel/BackgroundImpactAnalysis.git
    cd BackgroundImpactAnalysis
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Scripts Overview

### Main Script: `analysis/main.py`

This script is the entry point of the project. It orchestrates the creation of directories, counts the number of existing files, and processes images using a pre-trained DeepLabV3 model. The goal is to ensure that each class has a specified number of processed images.

#### Workflow:

1. **Directory Creation:**
    - The script first ensures that directories for storing masks and overlays are created if they do not already exist.
    - This is handled by the `create_directory` function.

2. **File Counting:**
    - The script counts the number of existing mask and overlay files in the respective directories using the `count_existing_files` function.
    - This helps in determining how many new images need to be processed.

3. **Model Loading:**
    - The DeepLabV3 model is loaded using the `load_model` function from the `analysis.seg_model_1` module.

4. **Processing Images:**
    - For each class specified, the script calculates the number of images that need to be processed.
    - It then calls the `process_and_save_images` function to generate the required number of masks and overlays.

#### Functions:

- `create_directory(path, dir_name)`: Ensures a directory exists or creates it if it does not. Returns a tuple indicating success and the directory path.
- `count_existing_files(directory)`: Counts the number of files in a specified directory. Returns the count of files.

### Dependencies:

- **DeepLabV3 Model**: The script uses a pre-trained DeepLabV3 model for image segmentation. Ensure the model and related weights are correctly set up in your environment.
- **OS Module**: Used for directory and file operations.

### Configuration:

- **Input Directory**: `./data/train/` - The directory where the input images are stored.
- **Output Directories**:
    - `./data/masks/` - The directory where the generated masks will be saved.
    - `./data/overlays/` - The directory where the generated overlays will be saved.
- **Classes**: A list of class labels to process.
- **Target Per Class**: The number of images to process per class.

### Segmentation Script: `analysis/seg_model_1.py`

This script provides the necessary functions to load the segmentation model, process images, and save the results. It includes functions for model loading, image transformation, segmentation, and saving the segmentation outputs as mask and overlay images.

#### Functions:

- `load_model()`: Loads the pre-trained DeepLabV3 ResNet101 model and sets it to evaluation mode.

- `transform_image(image_path)`: Transforms the input image into the appropriate format for the segmentation model.

- `segment(model, input_tensor)`: Performs segmentation on the transformed image using the provided model and returns the segmentation output tensor with class predictions.

- `apply_color_map_to_mask(predictions)`: Applies a color map to the segmentation mask to create a visually distinguishable mask image.

- `save_results(original_img, predictions, output_path_mask, output_path_overlay)`: Saves the segmentation results as mask and overlay images. The mask is saved as a binary mask, and the overlay is saved as a blended image of the original and the color-mapped mask.

- `process_and_save_images(input_dir, output_dir_mask, output_dir_overlay, model, images_to_process)`: Processes images in the input directory, performs segmentation, and saves the results in the specified output directories. The function ensures that a specified number of images are processed and saved.

### Mask Script: `analysis/analyze_masks.py`

This script analyzes mask images to identify the most common nonzero grayscale values for each class and lists exceptions where images do not contain this common value.

#### Workflow:

1. **Directory Traversal:**
    - Traverse the directory containing mask images for each class.

2. **Grayscale Value Analysis:**
    - Analyze each mask image to find unique nonzero grayscale values.
    - Use a counter to determine the most common nonzero grayscale value for each class.

3. **Exception Listing:**
    - Identify images that do not contain the most common grayscale value and list them as exceptions.

4. **Output Results:**
    - Print the most common grayscale value and its presence count for each class.
    - Print images that do not contain the most common grayscale value.

#### Functions:

- `analyze_masks_and_list_exceptions(masks_dir="data/masks")`: Analyzes mask images to find the most common nonzero grayscale value for each class and lists exceptions where images do not contain this common value.

#### Dependencies:

- **PIL**: Used for image loading and conversion to grayscale.
- **NumPy**: Used for array operations and unique value extraction.
- **os**: Used for directory and file path operations.
- **collections.Counter**: Used for counting occurrences of grayscale values.

### Configuration:

- **Masks Directory**: The default directory containing mask images is set to `"data/masks"`. This can be changed by passing a different directory path to the function.

This script helps in ensuring that the mask images are consistent in terms of the grayscale values they use for each class, and highlights any discrepancies for further inspection.

### BackgroundRemoval Script: `remove_bg.py`

This script removes the background from images using specified grayscale values from masks and saves the results.

#### Workflow:

1. **Directory Traversal:**
    - Traverse the input directory containing original images, the mask directory containing mask images, and create an output directory to save images with the background removed.

2. **Grayscale Value Retrieval:**
    - Retrieve the grayscale values for each class by calling the `analyze_masks_and_list_exceptions` function from the `analysis/analyze_masks.py` script.

3. **Background Removal:**
    - For each class, load the original image and its corresponding mask.
    - Use the specified grayscale value to create a mask for the object of interest.
    - Remove the background by creating a new image with the background set to black (or any other desired color) and the object of interest retained.
    - Save the resulting image in the output directory.

4. **Output Results:**
    - Print messages indicating the progress and any issues encountered, such as missing original images or already processed files.

#### Functions:

- `remove_background_and_save(input_dir, mask_dir, output_dir, grayscale_values)`: Removes the background from images using the specified grayscale values from masks and saves the results.

#### Dependencies:

- **PIL**: Used for image loading, conversion, and saving.
- **NumPy**: Used for array operations and masking.
- **os**: Used for directory and file path operations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values.

### Configuration:

- **Input Directory**: The directory containing the original images. Default is `"data/train"`.
- **Mask Directory**: The directory containing the mask images. Default is `"data/masks"`.
- **Output Directory**: The directory to save the images with the background removed. Default is `"data/modified/no_bg"`.

This script facilitates the removal of backgrounds from images based on mask information, providing a streamlined process for preparing modified images for further analysis or use.

### Color Script: `find_contrast_colors.py`

This script analyzes the colors in images within a directory to find dominant colors and determine high and low contrast colors for each class. The results are saved in a JSON file.

#### Workflow:

1. **Directory Traversal:**
    - Traverse the base directory containing images with backgrounds removed, and identify subdirectories for each class.

2. **Color Analysis:**
    - For each class directory, analyze the colors in the images to find dominant colors using the KMeans clustering algorithm.
    - Determine high and low contrast colors from the dominant colors.

3. **Save Results:**
    - Save the analyzed color data for each class in a JSON file.

#### Functions:

- `find_contrast_colors(colors)`: Finds high and low contrast colors given a set of dominant colors.
    - **Args:**
        - `colors (list)`: A list of RGB colors.
    - **Returns:**
        - `dict`: A dictionary with low and high contrast colors.

- `analyze_colors(directory)`: Analyzes the colors in the images within a directory to find dominant colors.
    - **Args:**
        - `directory (str)`: The path to the directory containing the images.
    - **Returns:**
        - `dict`: A dictionary with low and high contrast colors.

### Dependencies:

- **os**: Used for directory and file path operations.
- **json**: Used for saving the color analysis data to a JSON file.
- **NumPy**: Used for array operations and masking.
- **PIL**: Used for image loading.
- **scikit-learn (KMeans)**: Used for clustering the colors to find dominant colors.

### Configuration:

- **Base Directory**: The base directory containing the images with backgrounds removed. Default is `"data/modified/no_bg"`.

This script helps in analyzing and determining high and low contrast colors for images in different classes, providing valuable color information for further use or analysis.

### Foreground Script: `remove_foreground.py`

This script saves images with only the background, removing the foreground based on specified grayscale values from masks.

#### Workflow:

1. **Directory Traversal:**
    - Traverse the input directory containing original images and the mask directory containing mask images.
    - Create an output directory to save images with only the background.

2. **Grayscale Value Retrieval:**
    - Retrieve the grayscale values for each class by calling the `analyze_masks_and_list_exceptions` function from the `analysis/analyze_masks.py` script.

3. **Background Extraction:**
    - For each class, load the original image and its corresponding mask.
    - Use the specified grayscale value to create a mask for the background.
    - Save the resulting image, which only contains the background, in the output directory.

4. **Output Results:**
    - Print messages indicating the progress and any issues encountered, such as missing original images or already processed files.

#### Functions:

- `save_background_only(input_dir, mask_dir, output_dir, grayscale_values)`: Saves images with only the background, removing the foreground based on specified grayscale values from masks.
    - **Args:**
        - `input_dir (str)`: Directory containing the original images.
        - `mask_dir (str)`: Directory containing the mask images.
        - `output_dir (str)`: Directory to save the images with only the background.
        - `grayscale_values (dict)`: Dictionary mapping class names to grayscale values to be used for masking.

#### Dependencies:

- **PIL**: Used for image loading, conversion, and saving.
- **NumPy**: Used for array operations and masking.
- **os**: Used for directory and file path operations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values.

### Configuration:

- **Input Directory**: The directory containing the original images. Default is `"data/train"`.
- **Mask Directory**: The directory containing the mask images. Default is `"data/masks"`.
- **Output Directory**: The directory to save the images with only the background. Default is `"data/modified/no_foreground"`.

This script helps in extracting and saving the background from images based on mask information, providing a streamlined process for preparing modified images for further analysis or use.

### Contrast Script: `apply_contrast_background.py`

This script applies contrast backgrounds to images using masks and saves them. It leverages the most common grayscale values from masks and pre-defined contrast colors for each class.

#### Workflow:

1. **Directory Setup:**
    - Set up directories for input images (without background), masks, and output images with low and high contrast backgrounds.

2. **Grayscale Value and Contrast Color Retrieval:**
    - Retrieve the grayscale values for each class by calling the `analyze_masks_and_list_exceptions` function from the `analysis/analyze_masks.py` script.
    - Load the contrast colors for each class from a JSON file.

3. **Contrast Background Application:**
    - For each class, load the original image and its corresponding mask.
    - Use the specified grayscale value to create a mask for the object of interest.
    - Apply low and high contrast backgrounds to the image.
    - Save the resulting images in the respective directories.

4. **Output Results:**
    - Print messages indicating the progress and any issues encountered, such as missing mask files or already processed files.

#### Functions:

- `apply_contrast_background(input_dir, mask_dir, output_base_dir, contrast_colors, grayscale_values)`: Applies contrast backgrounds to images using masks and saves them.
    - **Args:**
        - `input_dir (str)`: Directory containing the original images without background.
        - `mask_dir (str)`: Directory containing the mask images.
        - `output_base_dir (str)`: Base directory to save the processed images with contrast backgrounds.
        - `contrast_colors (dict)`: Dictionary mapping class names to their low and high contrast colors.
        - `grayscale_values (dict)`: Dictionary mapping class names to their most common nonzero grayscale values in masks.

#### Dependencies:

- **PIL**: Used for image loading, conversion, and saving.
- **NumPy**: Used for array operations and masking.
- **os**: Used for directory and file path operations.
- **json**: Used for loading contrast color configurations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values.

### Configuration:

- **Input Directory**: The directory containing the original images without background. Default is `"data/modified/no_bg"`.
- **Mask Directory**: The directory containing the mask images. Default is `"data/masks"`.
- **Output Directory**: The base directory to save the processed images with contrast backgrounds. Default is `"data/modified"`.
- **Color Configuration Path**: Path to the JSON file containing contrast color configurations. Default is `"class_colors.json"`.

This script facilitates the application of contrast backgrounds to images based on mask information and pre-defined color configurations, providing a streamlined process for generating visually distinct images for each class.

### Scenic Script: `scenic_bg.py`

This script applies different scenic backgrounds to images using masks, skipping images that are already processed.

#### Workflow:

1. **Directory Setup:**
    - Set up directories for input images, mask images, output images with applied scenic backgrounds, and background images.

2. **Grayscale Value Retrieval:**
    - Retrieve the grayscale values for each class by calling the `analyze_masks_and_list_exceptions` function from the `analysis/analyze_masks.py` script.

3. **Scenic Background Application:**
    - For each class, load the original image and its corresponding mask.
    - Use the specified grayscale value to create a mask for the object of interest.
    - Apply different scenic backgrounds to the image.
    - Save the resulting images in the respective directories for each scenario.

4. **Output Results:**
    - Print messages indicating the progress and any issues encountered, such as missing mask files or already processed files.

#### Functions:

- `apply_scenic_backgrounds(input_dir, mask_dir, output_dir, background_dir, backgrounds_info, grayscale_values)`: Applies different scenic backgrounds to images using masks and saves them.
    - **Args:**
        - `input_dir (str)`: Directory containing the original images.
        - `mask_dir (str)`: Directory containing the mask images.
        - `output_dir (str)`: Directory to save the images with applied scenic backgrounds.
        - `background_dir (str)`: Directory containing the background images.
        - `backgrounds_info (dict)`: Dictionary mapping scenario names to background image filenames.
        - `grayscale_values (dict)`: Dictionary mapping class names to their most common nonzero grayscale values in masks.

#### Dependencies:

- **PIL**: Used for image loading, conversion, and saving.
- **NumPy**: Used for array operations and masking.
- **os**: Used for directory and file path operations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values.

### Configuration:

- **Input Directory**: The directory containing the original images. Default is `"data/train"`.
- **Mask Directory**: The directory containing the mask images. Default is `"data/masks"`.
- **Output Directory**: The directory to save the images with applied scenic backgrounds. Default is `"data/modified"`.
- **Background Directory**: The directory containing the background images. Default is `"data/scenarios"`.
- **Backgrounds Info**: Dictionary mapping scenario names to background image filenames. Default is:
    ```python
    backgrounds_info = {
        "city": "city.jpg",
        "jungle": "jungle.jpg",
        "desert": "desert.jpg",
        "water": "water.jpg",
        "sky": "sky.jpg",
        "indoor": "indoor.jpg",
        "mountain": "mountain.jpg",
        "snow": "snow.jpg",
    }
    ```

This script helps in applying various scenic backgrounds to images based on mask information and pre-defined background scenarios, providing a streamlined process for generating visually distinct images for each class.

### Resnet Script: `resnet.py`

This script compares original images with their modified versions using a pre-trained ResNet model and saves the top-5 class predictions and their confidence scores to a CSV file.

#### Workflow:

1. **Model Loading:**
    - Load the pre-trained ResNet model and set it to evaluation mode on the specified device (CPU or GPU).

2. **Image Transformation:**
    - Define the necessary image transformations to preprocess the images for the ResNet model.

3. **Image Prediction:**
    - For each image, predict the top-5 class probabilities using the pre-trained ResNet model.

4. **Comparison and Results Saving:**
    - Compare the predictions of the original images with their modified versions.
    - Save the top-5 predictions and confidence scores to a CSV file.

#### Functions:

- `load_model(device)`: Loads the pre-trained ResNet model and sets it to evaluation mode on the specified device.
    - **Args:**
        - `device (torch.device)`: The device to load the model on.
    - **Returns:**
        - `torch.nn.Module`: The loaded ResNet model.

- `get_transform()`: Defines the image transformation.
    - **Returns:**
        - `torchvision.transforms.Compose`: The composed image transformations.

- `predict_top5(image_path, model, device, transform)`: Predicts the top-5 class probabilities for an image.
    - **Args:**
        - `image_path (str)`: The path to the image.
        - `model (torch.nn.Module)`: The pre-trained model.
        - `device (torch.device)`: The device to perform inference on.
        - `transform (torchvision.transforms.Compose)`: The image transformations.
    - **Returns:**
        - `list`: A list of dictionaries with class indices and confidence scores.

- `compare_images_and_save_results(original_dir, modifications_root_dir, model, device, transform, output_csv_path, exceptions_dic, target_per_class=1000)`: Compares original images with modified versions and saves the top-5 predictions to a CSV file.
    - **Args:**
        - `original_dir (str)`: Directory containing the original images.
        - `modifications_root_dir (str)`: Root directory containing the modified images.
        - `model (torch.nn.Module)`: The pre-trained model.
        - `device (torch.device)`: The device to perform inference on.
        - `transform (torchvision.transforms.Compose)`: The image transformations.
        - `output_csv_path (str)`: The path to save the CSV file with results.
        - `exceptions_dic (dict)`: Dictionary of images to exclude from processing.
        - `target_per_class (int)`: The target number of images to process per class.

#### Dependencies:

- **torch**: Used for loading the pre-trained model and performing inference.
- **torchvision**: Used for model loading and image transformations.
- **PIL**: Used for image loading and conversion.
- **pandas**: Used for saving the results to a CSV file.
- **os**: Used for directory and file path operations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values and exceptions.

### Configuration:

- **Device**: The device to perform inference on, either CPU or GPU.
- **Original Images Directory**: The directory containing the original images. Default is `"data/train"`.
- **Modifications Root Directory**: The root directory containing the modified images. Default is `"data/modified"`.
- **Output CSV**: The path to save the CSV file with the results. Default is `"image_confidence_scores_resnet.csv"`.

This script helps in comparing original and modified images, providing insights into how modifications affect the classification results, and saves the results for further analysis.

### Convnext Script: `convnext.py`

This script compares original images with their modified versions using a pre-trained ConvNeXt model and saves the top-5 class predictions and their confidence scores to a CSV file.

#### Workflow:

1. **Model Loading:**
    - Load the pre-trained ConvNeXt model and set it to evaluation mode on the specified device (CPU or GPU).

2. **Image Transformation:**
    - Define the necessary image transformations to preprocess the images for the ConvNeXt model.

3. **Image Prediction:**
    - For each image, predict the top-5 class probabilities using the pre-trained ConvNeXt model.

4. **Comparison and Results Saving:**
    - Compare the predictions of the original images with their modified versions.
    - Save the top-5 predictions and confidence scores to a CSV file.

#### Functions:

- `load_model(device)`: Loads the pre-trained ConvNeXt model and sets it to evaluation mode on the specified device.
    - **Args:**
        - `device (torch.device)`: The device to load the model on.
    - **Returns:**
        - `torch.nn.Module`: The loaded ConvNeXt model.

- `get_transform()`: Defines the image transformation.
    - **Returns:**
        - `torchvision.transforms.Compose`: The composed image transformations.

- `predict_top5(image_path, model, device, transform)`: Predicts the top-5 class probabilities for an image.
    - **Args:**
        - `image_path (str)`: The path to the image.
        - `model (torch.nn.Module)`: The pre-trained model.
        - `device (torch.device)`: The device to perform inference on.
        - `transform (torchvision.transforms.Compose)`: The image transformations.
    - **Returns:**
        - `list`: A list of dictionaries with class indices and confidence scores.

- `compare_images_and_save_results(original_dir, modifications_root_dir, model, device, transform, output_csv_path, exceptions_dic, target_per_class=1000)`: Compares original images with modified versions and saves the top-5 predictions to a CSV file.
    - **Args:**
        - `original_dir (str)`: Directory containing the original images.
        - `modifications_root_dir (str)`: Root directory containing the modified images.
        - `model (torch.nn.Module)`: The pre-trained model.
        - `device (torch.device)`: The device to perform inference on.
        - `transform (torchvision.transforms.Compose)`: The image transformations.
        - `output_csv_path (str)`: The path to save the CSV file with results.
        - `exceptions_dic (dict)`: Dictionary of images to exclude from processing.
        - `target_per_class (int)`: The target number of images to process per class.

#### Dependencies:

- **torch**: Used for loading the pre-trained model and performing inference.
- **torchvision**: Used for model loading and image transformations.
- **PIL**: Used for image loading and conversion.
- **pandas**: Used for saving the results to a CSV file.
- **os**: Used for directory and file path operations.
- **analyze_masks_and_list_exceptions**: Function from the `analysis/analyze_masks.py` script to retrieve the most common grayscale values and exceptions.

### Configuration:

- **Device**: The device to perform inference on, either CPU or GPU.
- **Original Images Directory**: The directory containing the original images. Default is `"data/train"`.
- **Modifications Root Directory**: The root directory containing the modified images. Default is `"data/modified"`.
- **Output CSV**: The path to save the CSV file with the results. Default is `"image_confidence_scores_convnext.csv"`.

This script helps in comparing original and modified images, providing insights into how modifications affect the classification results, and saves the results for further analysis.
