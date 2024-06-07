import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

def load_model(device):
    # Load the pre-trained ConvNeXt model
    model = models.convnext_base(pretrained=True).to(device)
    model.eval()
    return model

def get_transform():
    # Define the image transformation
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_top5(image_path, model, device, transform):
    try:
        with Image.open(image_path).convert("RGB") as img:
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            top5_conf, top5_class = torch.topk(probs, 5)
            return [{int(class_idx): float(conf)} for conf, class_idx in zip(top5_conf, top5_class)]
    except (IOError, OSError) as e:
        print(f"Warning: Could not process image {image_path}: {e}")
        return None

def compare_images_and_save_results(original_dir, modifications_root_dir, model, device, transform, output_csv_path):
    results = []
    print("Starting image comparison...")

    # Get all modification types
    modification_types = [d for d in os.listdir(modifications_root_dir) if os.path.isdir(os.path.join(modifications_root_dir, d))]
    print(f"Found modification types: {modification_types}")

    # Collect information on available images for each modification
    image_paths = {}

    # Gather all available images and their paths from modifications
    for modification_type in modification_types:
        mod_path = os.path.join(modifications_root_dir, modification_type)
        for class_name in os.listdir(mod_path):
            class_path = os.path.join(mod_path, class_name)
            for image_name in os.listdir(class_path):
                base_filename, _ = os.path.splitext(image_name)
                if base_filename not in image_paths:
                    image_paths[base_filename] = {}
                image_paths[base_filename][modification_type] = os.path.join(class_path, image_name)
                image_paths[base_filename]['class_name'] = class_name  # Keep track of class name

    total_images = len(image_paths)
    processed_count = 0

    # Check against original images and compile results if all modifications exist
    for base_filename, paths in image_paths.items():
        processed_count += 1
        print(f"Processing image {processed_count} of {total_images} ({base_filename})")

        if len(paths) - 1 != len(modification_types):  # Subtract 1 to exclude 'class_name'
            continue  # Skip if not all modifications are available

        original_image_path = os.path.join(original_dir, paths['class_name'], f"{base_filename}.JPEG")
        if not os.path.exists(original_image_path):
            continue

        # Attempt to get top-5 predictions for the original image
        original_top5 = predict_top5(original_image_path, model, device, transform)
        if original_top5 is None:
            continue

        # Prepare results entry
        result = {
            "id": base_filename,
            "picture_name": f"{base_filename}.JPEG",
            "original_confidence": original_top5,
        }

        # Process each modification
        all_mods_valid = True
        for mod_type in modification_types:
            mod_image_path = paths.get(mod_type)
            if mod_image_path:
                mod_top5 = predict_top5(mod_image_path, model, device, transform)
                if mod_top5 is None:
                    all_mods_valid = False
                    break
                result[f"{mod_type}_confidence"] = mod_top5
            else:
                all_mods_valid = False
                break

        if all_mods_valid:
            results.append(result)

    # Save results to a CSV file
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print("Results successfully saved to CSV.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = get_transform()

    original_images_dir = "data/train"
    modifications_root_dir = "data/modified"
    output_csv = "image_confidence_scores_convnext.csv"

    compare_images_and_save_results(
        original_images_dir, modifications_root_dir, model, device, transform, output_csv
    )
