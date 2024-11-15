import os
import json

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToPILImage


class SarcasmDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        list_id,
        processed_dir=None,
        image_transform=None,
        text_transform=None,
        target_transform=None,
        mode="train",
        submit=False,
    ):
        """
        Parameters:
        - annotations_file: Path to the annotations JSON file.
        - img_dir: Directory with the original images.
        - list_id: List of image IDs.
        - processed_dir: Directory where preprocessed data (masked images, OCR text) is stored.
        - image_transform: Transformations applied to images (optional).
        - text_transform: Transformations applied to text (optional).
        - target_transform: Transformations applied to the target labels (optional).
        - mode: 'train' or 'inference', determines behavior for missing preprocessed data.
        """
        with open(annotations_file, "rb") as f:
            dict_annot = json.load(f)

        self.list_id = list_id
        self.img_labels = [dict_annot[key] for key in self.list_id]

        self.submit = submit
        self.img_dir = img_dir
        self.processed_dir = (
            processed_dir  # Directory where preprocessed data is stored
        )
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.mode = mode

        self.labels_2_id = {
            "not-sarcasm": 0,
            "text-sarcasm": 1,
            "image-sarcasm": 2,
            "multi-sarcasm": 3,
        }

        self.id_2_labels = {
            0: "not-sarcasm",
            1: "text-sarcasm",
            2: "image-sarcasm",
            3: "multi-sarcasm",
        }

    def get_all_labels(self):
        return [self.labels_2_id[img_label["label"]] for img_label in self.img_labels]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Image file path
        image_filename = self.img_labels[idx]["image"]
        image_name_without_ext = image_filename.split(".")[0]
        img_path = os.path.join(self.img_dir, image_filename)
        _id = self.list_id[idx]

        # Preprocessed paths
        masked_image_path = None
        extracted_text_path = None
        if self.processed_dir:
            masked_image_path = os.path.join(
                self.processed_dir, "images", f"{image_name_without_ext}_masked.png"
            )
            extracted_text_path = os.path.join(
                self.processed_dir, "text", f"{image_name_without_ext}.txt"
            )

        image = None
        image_text = None

        # Check if preprocessed data exists
        if self.processed_dir and os.path.exists(extracted_text_path):
            # Load preprocessed masked image and OCR text
            #             image = read_image(masked_image_path)
            image = read_image(img_path)
            with open(extracted_text_path, "r") as text_file:
                image_text = text_file.read().strip()  # Load pre-extracted OCR text
        else:
            # No preprocessed data available, apply OCRMaskTransform
            image = read_image(img_path)
            if self.image_transform:
                image, image_text = self.image_transform(image)

            # Save the masked image and OCR text (only during training)
            if self.processed_dir and self.mode == "train":
                os.makedirs(os.path.join(self.processed_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.processed_dir, "text"), exist_ok=True)

                # Save masked image
                masked_image_pil = ToPILImage()(image)
                masked_image_pil.save(masked_image_path)

                # Save extracted text
                with open(extracted_text_path, "w") as text_file:
                    text_file.write(
                        image_text or ""
                    )  # Handle missing text with empty string

        # Get original caption and label
        caption = self.img_labels[idx]["caption"]
        if self.submit:
            label = -1
        else:
            label = self.labels_2_id[self.img_labels[idx]["label"]]

        # Apply text and target transforms if necessary
        if self.text_transform:
            caption = self.text_transform(caption)
        if self.target_transform and label != -1:
            label = self.target_transform(label)

        return (image, image_text, caption), label, _id


class SarcasmDatasetForFinetuneText(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        list_id,
        processed_dir=None,
        image_transform=None,
        text_transform=None,
        target_transform=None,
        mode="train",
        submit=False,
        augmentations=None,
    ):
        supported_labels = ("text-sarcasm", "image-sarcasm")
        with open(annotations_file, "rb") as f:
            dict_annot = json.load(f)
        self.list_id = list_id
        self.img_labels = [
            dict_annot[key]
            for key in self.list_id
            if dict_annot[key]["label"] in supported_labels
        ]
        self.submit = submit
        self.img_dir = img_dir
        self.processed_dir = processed_dir
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.mode = mode
        self.augmentations = augmentations  # Text augmentations
        self.labels_2_id = {
            "not-sarcasm": 0,
            "text-sarcasm": 1,
            "image-sarcasm": 2,
            "multi-sarcasm": 3,
        }

    def __getitem__(self, idx):
        image_filename = self.img_labels[idx]["image"]
        image_name_without_ext = image_filename.split(".")[0]
        # img_path = os.path.join(self.img_dir, image_filename)
        # _id = self.list_id[idx]

        if self.processed_dir:
            extracted_text_path = os.path.join(
                self.processed_dir, "text", f"{image_name_without_ext}.txt"
            )
        caption = self.img_labels[idx]["caption"]

        # Load OCR extracted text if available
        if self.processed_dir and os.path.exists(extracted_text_path):
            with open(extracted_text_path, "r") as text_file:
                image_text = text_file.read().strip()
        else:
            image_text = ""

        # Apply augmentations
        if self.augmentations:
            caption = self.augmentations(caption)
            image_text = self.augmentations(image_text)

        # Labels for sarcasm
        true_label = self.img_labels[idx]["label"]
        caption_label = 1 if true_label == "text-sarcasm" else 0
        ocr_label = 1 if true_label == "image-sarcasm" else 0

        return (caption, image_text), (caption_label, ocr_label)

    def __len__(self):
        return len(self.img_labels)
