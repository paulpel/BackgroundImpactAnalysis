# IMAGE BACKGROUND ANALYSIS

DESCRIPTION TO DO

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [Main Script](#main-script)


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

## Usage

To run scripts use following command

```bash
python -m analysis.script_name
```

## Scripts Overview

### Main Script: `main.py`

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

### Execution:

To run the script, execute the following command in your terminal:
```bash
python -m analysis.main


