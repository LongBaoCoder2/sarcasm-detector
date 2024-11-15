from typing import List

import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torchvision.io import read_image
from torchvision.transforms import ToPILImage, ToTensor

import easyocr

from dataset import SarcasmDataset


class OCRMaskTransform:
    def __init__(
        self, lang: str | List[str] = ["vi", "en"], mask_type="solid", mask_value=0
    ):
        """
        Initializes the OCR mask transform.

        Parameters:
        - lang: list of languages to pass to easyocr Reader.
        - mask_type: 'solid' for a solid color mask, 'blur' for Gaussian blur masking.
        - mask_value: Value for the solid mask (default is black, i.e., 0).
        """
        if isinstance(lang, str):
            self.lang = [lang]
        else:
            self.lang = lang
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.reader = easyocr.Reader(
            self.lang
        )  # Initialize easyocr Reader with language

    def __call__(self, image):
        # Convert the image from torch Tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            image = ToPILImage()(image)
        if isinstance(image, str):
            image = Image.open(image)

        # Convert image to OpenCV format for text detection
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # OCR to extract text and its bounding boxes using easyocr
        ocr_result = self.reader.readtext(open_cv_image)

        extracted_text = " ".join([res[1] for res in ocr_result]).strip() or None

        if extracted_text:
            # Iterate over detected text boxes and apply the mask
            for res in ocr_result:
                bbox, text, _ = res  # bbox contains the coordinates of the text box

                # easyocr returns the bounding box as a list of four points
                # Convert bounding box into a rectangular area
                x_min = int(min([point[0] for point in bbox]))
                y_min = int(min([point[1] for point in bbox]))
                x_max = int(max([point[0] for point in bbox]))
                y_max = int(max([point[1] for point in bbox]))

                # Apply mask based on the selected mask type
                if self.mask_type == "solid":
                    # Replace the text area with a solid color (black by default)
                    open_cv_image[y_min:y_max, x_min:x_max] = self.mask_value
                elif self.mask_type == "blur":
                    # Apply Gaussian blur to the text area
                    sub_img = open_cv_image[y_min:y_max, x_min:x_max]
                    blurred = cv2.GaussianBlur(sub_img, (15, 15), 0)
                    open_cv_image[y_min:y_max, x_min:x_max] = blurred

        # Convert back to PIL image
        masked_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))

        # Convert back to Tensor for PyTorch model usage
        masked_image_tensor = ToTensor()(masked_image)

        return masked_image_tensor, extracted_text


def preprocess_images(dataset: SarcasmDataset, output_dir: str):
    os.makedirs(
        os.path.join(output_dir, "images"), exist_ok=True
    )  # Directory for masked images
    os.makedirs(
        os.path.join(output_dir, "text"), exist_ok=True
    )  # Directory for extracted texts

    for idx in tqdm(range(len(dataset))):
        image_filename = dataset.img_labels[idx]["image"]
        image_name_without_ext = image_filename.split(".")[0]

        # masked_image_path = os.path.join(
        #     output_dir, "images", f"{image_name_without_ext}_masked.png"
        # )
        text_file_path = os.path.join(
            output_dir, "text", f"{image_name_without_ext}.txt"
        )

        if os.path.exists(text_file_path):
            continue

        img_path = os.path.join(dataset.img_dir, image_filename)
        image = read_image(img_path)
        masked_image, extracted_text = dataset.image_transform(image)

        #         if not os.path.exists(masked_image_path):
        #             masked_image_pil = ToPILImage()(masked_image)
        #             masked_image_pil.save(masked_image_path)

        # Save the extracted text if it doesn't exist
        if not os.path.exists(text_file_path):
            with open(text_file_path, "w") as text_file:
                if extracted_text:
                    text_file.write(extracted_text)
                else:
                    text_file.write("")  # Write an empty string if no text is detected
