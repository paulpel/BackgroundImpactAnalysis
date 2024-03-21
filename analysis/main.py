from analysis.seg_model_1 import process_and_save_images, load_model
import os


def create_directory(path, dir_name):
    directory_path = os.path.join(path, dir_name)

    try:
        os.makedirs(directory_path, exist_ok=True)
        return (True, directory_path)
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")
        return (False, directory_path)


if __name__ == "__main__":
    input_directory = './data/train/'
    output_mask_directory = './data/masks/'
    output_overlay_directory = './data/overlays/'

    deeplabv3_model = load_model()

    classes = [
        'n01534433', 'n02106662', 'n02124075', 'n02415577', 'n02105641',
        'n01833805', 'n01558993', 'n02412080', 'n02107574', 'n02123394'
    ]

    for c in classes:
        _, mask_dir = create_directory(output_mask_directory, c)
        _, overlay_dir = create_directory(output_overlay_directory, c)
        input_dir = os.path.join(input_directory, c)
        process_and_save_images(
            input_dir, mask_dir, overlay_dir, deeplabv3_model, 100)

