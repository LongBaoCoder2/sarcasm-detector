import torch
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt


def visualize_noisy_image(original_image, noisy_image_tensor, extracted_text=None):
    """
    Visualizes the original and noisy image side by side and displays the extracted text.

    Parameters:
    - original_image: PIL Image or torch Tensor (original image before noise)
    - noisy_image_tensor: torch Tensor (noisy image after transformation)
    - extracted_text: str or None (text extracted via OCR)
    """

    # Convert the noisy image tensor back to PIL Image
    if isinstance(noisy_image_tensor, torch.Tensor):
        noisy_image = ToPILImage()(noisy_image_tensor)
    else:
        noisy_image = noisy_image_tensor

    # Convert the original image tensor back to PIL Image (if needed)
    if isinstance(original_image, torch.Tensor):
        original_image = ToPILImage()(original_image)

    # Create a figure to display the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")  # Turn off axis

    # Noisy Image
    axes[1].imshow(noisy_image)
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")  # Turn off axis

    # Show the extracted text below the images (if any)
    if extracted_text:
        plt.figtext(
            0.5,
            0.01,
            f"Extracted Text: {extracted_text}",
            ha="center",
            fontsize=12,
            color="green",
        )
    else:
        plt.figtext(
            0.5, 0.01, "Extracted Text: None", ha="center", fontsize=12, color="red"
        )

    # Display the images
    plt.show()
