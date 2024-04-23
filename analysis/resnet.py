import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import csv


def load_model(device):
    # Load the pre-trained ResNet model
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    return model


def get_transform():
    # Define the image transformation
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_top5(image_path, model, device, transform):
    # Open the image file
    with Image.open(image_path).convert("RGB") as img:  # Ensure the image is in RGB
        img_t = transform(img).unsqueeze(0).to(device)

        # Get the predictions
        with torch.no_grad():
            outputs = model(img_t)

        # Get the top 5 predictions and their associated confidence scores
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        top5_conf, top5_class = torch.topk(probs, 5)

        return [
            {int(class_idx): float(conf)}
            for conf, class_idx in zip(top5_conf, top5_class)
        ]


def compare_images_and_save_results(
    original_dir, modified_dir, model, device, transform, output_csv_path
):
    results = []
    for class_name in os.listdir(modified_dir):
        original_class_dir = os.path.join(original_dir, class_name)
        modified_class_dir = os.path.join(modified_dir, class_name)

        for modified_image_name in os.listdir(modified_class_dir):
            # The original image will have the same base name but a different extension
            base_filename = os.path.splitext(modified_image_name)[0]
            original_image_name = f"{base_filename}.JPEG"

            original_image_path = os.path.join(original_class_dir, original_image_name)
            modified_image_path = os.path.join(modified_class_dir, modified_image_name)

            # If original image doesn't exist, skip it
            if not os.path.exists(original_image_path):
                continue

            original_top5 = predict_top5(original_image_path, model, device, transform)
            modified_top5 = predict_top5(modified_image_path, model, device, transform)

            results.append(
                {
                    "id": base_filename,
                    "picture_name": modified_image_name,
                    "original_confidence": original_top5,
                    "removed_bg_confidence": modified_top5,
                }
            )

    # Save results to a CSV file
    pd.DataFrame(results).to_csv(output_csv_path, index=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
transform = get_transform()

original_images_dir = "data/train"
removed_bg_images_dir = "data/modified/no_bg"
output_csv = "image_confidence_scores.csv"

compare_images_and_save_results(
    original_images_dir, removed_bg_images_dir, model, device, transform, output_csv
)
