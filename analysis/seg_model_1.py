import os
import torchvision.models.segmentation as segmentation
from torchvision import transforms
from PIL import Image
import torch


def load_model():
    """
    Load the pre-trained DeepLabV3 ResNet101 model.

    Returns:
        torch.nn.Module: The pre-trained DeepLabV3 ResNet101 model in evaluation mode.
    """
    model = segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model


def transform_image(image_path):
    """
    Transform the input image for the segmentation model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    input_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(input_image)


def segment(model, input_tensor):
    """
    Perform segmentation on the input tensor using the provided model.

    Args:
        model (torch.nn.Module): The segmentation model.
        input_tensor (torch.Tensor): The transformed image tensor.

    Returns:
        torch.Tensor: The segmentation output tensor with class predictions.
    """
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)["out"][0]
    return output.argmax(0)


def apply_color_map_to_mask(predictions):
    """
    Apply a color map to the segmentation mask.

    Args:
        predictions (torch.Tensor): The segmentation output tensor with class predictions.

    Returns:
        PIL.Image: The color-mapped segmentation mask image.
    """
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(predictions.byte().cpu().numpy())
    r.putpalette(colors)
    return r


def save_results(original_img, predictions, output_path_mask, output_path_overlay):
    """
    Save the segmentation results as mask and overlay images.

    Args:
        original_img (PIL.Image): The original input image.
        predictions (torch.Tensor): The segmentation output tensor with class predictions.
        output_path_mask (str): Path to save the mask image.
        output_path_overlay (str): Path to save the overlay image.
    """
    color_mask = apply_color_map_to_mask(predictions)

    color_mask = color_mask.resize(original_img.size, resample=Image.NEAREST).convert(
        "RGBA"
    )

    mask = Image.fromarray(predictions.byte().cpu().numpy())
    mask = mask.resize(original_img.size, resample=Image.NEAREST)
    mask.save(output_path_mask)

    overlay = Image.blend(original_img.convert("RGBA"), color_mask, alpha=0.5)
    overlay.save(output_path_overlay)


def process_and_save_images(
    input_dir, output_dir_mask, output_dir_overlay, model, images_to_process
):
    """
    Process and save segmentation results for images in a directory.

    Args:
        input_dir (str): Directory containing input images.
        output_dir_mask (str): Directory to save the mask images.
        output_dir_overlay (str): Directory to save the overlay images.
        model (torch.nn.Module): The segmentation model.
        images_to_process (int): Number of images to process.

    Prints:
        str: The number of images processed and saved.
    """
    processed_count = 0

    for image_name in os.listdir(input_dir):
        if processed_count >= images_to_process:
            break

        base_filename, _ = os.path.splitext(image_name)
        output_path_mask = os.path.join(output_dir_mask, f"{base_filename}.png")
        output_path_overlay = os.path.join(output_dir_overlay, f"{base_filename}.png")

        if os.path.exists(output_path_mask) and os.path.exists(output_path_overlay):
            continue

        image_path = os.path.join(input_dir, image_name)
        original_image = Image.open(image_path).convert("RGB")

        input_tensor = transform_image(image_path)
        output_predictions = segment(model, input_tensor)

        save_results(
            original_image, output_predictions, output_path_mask, output_path_overlay
        )

        processed_count += 1

    print(
        f"Processed and saved {processed_count} additional images for {os.path.basename(input_dir)}."
    )
