import argparse
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from dataset import SarcasmDataset
from models.model import SarcasmModel
from utils.trainer import train_model


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sarcasm detection model using multimodal embeddings."
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use GPU for training if available."
    )
    parser.add_argument(
        "--annotations_path", type=str, required=True, help="Path to the metadata file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to the directory for processed data.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="uitnlp/visobert",
        help="HuggingFace model for text embeddings.",
    )
    parser.add_argument(
        "--image_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model for image embeddings.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.3, help="Proportion of data for testing."
    )
    return parser.parse_args()


# Get transforms
def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomErasing(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, test_transforms


def get_freezed_embeders(args):
    image_embeder = CLIPModel.from_pretrained(args.image_model)
    text_embeder = AutoModel.from_pretrained(args.text_model)

    for param in image_embeder.parameters():
        param.requires_grad = False
    for param in text_embeder.parameters():
        param.requires_grad = False

    return image_embeder, text_embeder


# Main function
def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load models
    image_processor = CLIPProcessor.from_pretrained(args.image_model)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    image_clip_processor = lambda images: image_processor(
        images=images, return_tensors="pt"
    )

    def text_sbert_processor(texts):
        tokenized = text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=256
        )
        input_ids = tokenized["input_ids"]  # Get the token IDs
        # Convert IDs back to tokens
        tokens = [text_tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        return tokenized, tokens

    def collate_fn(batch):
        images, image_texts, texts, ids = [], [], [], []
        labels = []
        for (image, image_text, text), label, _id in batch:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            images.append(to_pil_image(image))
            image_texts.append(image_text)
            texts.append(text)
            labels.append(label)
            ids.append(_id)

        # Process images and texts using CLIPProcessor
        image_inputs = image_clip_processor(images)

        encoded_image_texts, _ = text_sbert_processor(image_texts)
        encoded_texts, _ = text_sbert_processor(texts)
        labels = torch.tensor(labels)

        return (image_inputs, encoded_image_texts, encoded_texts), labels, ids

    image_embeder, text_embeder = get_freezed_embeders(args)

    model = SarcasmModel(
        text_embeder,
        image_embeder,
        dropout=0.4,
        text_dropout=0.2,
        image_dropout=0.45,
        fusion_dim=768,
        rank=32,
        output_dim=4,
    ).to(device)

    # Get transforms and datasets
    train_transforms, test_transforms = get_transforms()
    metadata = SarcasmDataset.load_metadata(args.annotations_path)

    list_id = list(metadata.keys())
    labels = [SarcasmDataset.get_label(metadata[img_id]) for img_id in list_id]

    strat_split = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_ratio, random_state=42
    )
    train_idx, test_idx = next(strat_split.split(list_id, labels))

    train_ids = [list_id[i] for i in train_idx]
    test_ids = [list_id[i] for i in test_idx]

    train_dataset = SarcasmDataset(
        args.annotations_path,
        args.image_dir,
        train_ids,
        args.processed_dir,
        train_transforms,
        mode="train",
    )
    test_dataset = SarcasmDataset(
        args.annotations_path,
        args.image_dir,
        test_ids,
        args.processed_dir,
        test_transforms,
        mode="test",
    )

    # Train model
    train_model(
        model,
        train_dataset,
        test_dataset,
        collate_fn,
        args.batch_size,
        args.num_epochs,
        args.learning_rate,
        args.patience,
        device,
    )


if __name__ == "__main__":
    main()
