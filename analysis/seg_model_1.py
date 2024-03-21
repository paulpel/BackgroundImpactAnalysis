import os
import torchvision.models.segmentation as segmentation
from torchvision import transforms
from PIL import Image
import torch

def load_model():
    model = segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def transform_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(520),  # Resize the image to 520 pixels on the smaller side
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(input_image)

def segment(model, input_tensor):
    input_batch = input_tensor.unsqueeze(0)  # Create a batch by adding a batch dimension
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    return output.argmax(0)

def apply_color_map_to_mask(predictions):
    # Define the colormap to apply
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Apply colormap to predictions
    r = Image.fromarray(predictions.byte().cpu().numpy())
    r.putpalette(colors)
    return r

def save_results(original_img, predictions, output_path_mask, output_path_overlay):
    # Apply a color map to the mask predictions
    color_mask = apply_color_map_to_mask(predictions)
    
    # Resize the color mask to match the original image
    color_mask = color_mask.resize(original_img.size, resample=Image.NEAREST).convert('RGBA')

    # Save the grayscale mask
    mask = Image.fromarray(predictions.byte().cpu().numpy())
    mask = mask.resize(original_img.size, resample=Image.NEAREST)
    mask.save(output_path_mask)

    # Create an overlay image by blending the original image with the color mask
    overlay = Image.blend(original_img.convert('RGBA'), color_mask, alpha=0.5)
    # Save the overlay
    overlay.save(output_path_overlay)

def process_and_save_images(input_dir, output_dir_mask, output_dir_overlay, model, counter=None):
    c = 0
    for image_name in os.listdir(input_dir):
        if counter is not None and c >= counter:
            break

        image_path = os.path.join(input_dir, image_name)
        original_image = Image.open(image_path).convert("RGB")
        
        input_tensor = transform_image(image_path)
        output_predictions = segment(model, input_tensor)
        
        # Define the output paths
        base_filename, _ = os.path.splitext(image_name)
        output_path_mask = os.path.join(output_dir_mask, f"{base_filename}.png")
        output_path_overlay = os.path.join(output_dir_overlay, f"{base_filename}.png")
        
        # Save the mask and the overlay image
        save_results(original_image, output_predictions, output_path_mask, output_path_overlay)

        c += 1

