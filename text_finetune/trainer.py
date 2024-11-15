import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from dataset import SarcasmDatasetForFinetuneText
from text_finetune.text_detector import SarcasmClassifier


# Training function
def train_model(args):
    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    text_embeder = AutoModel.from_pretrained(args.model_name)
    model = SarcasmClassifier(text_embeder).to(args.device)

    # Prepare DataLoaders
    train_dataset = SarcasmDatasetForFinetuneText(
        annotations_file=args.metadata_path,
        img_dir=args.image_dir,
        list_id=args.train_list_id,
        processed_dir=args.processed_dir,
    )

    test_dataset = SarcasmDatasetForFinetuneText(
        annotations_file=args.metadata_path,
        img_dir=args.image_dir,
        list_id=args.test_list_id,
        processed_dir=args.processed_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss function with class weights
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(
        args.device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct_captions = 0
        correct_ocr = 0
        total = 0

        for batch_idx, (
            (captions, ocr_texts),
            (caption_labels, ocr_labels),
        ) in enumerate(train_loader):
            optimizer.zero_grad()

            # Tokenize input
            caption_inputs = tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(args.device)
            ocr_inputs = tokenizer(
                ocr_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(args.device)

            # Prepare labels
            caption_labels = torch.tensor(caption_labels).to(args.device)
            ocr_labels = torch.tensor(ocr_labels).to(args.device)

            # Forward pass
            caption_pred, ocr_pred = model(caption_inputs, ocr_inputs)

            # Compute loss
            caption_loss = criterion(caption_pred, caption_labels)
            ocr_loss = criterion(ocr_pred, ocr_labels)
            loss = caption_loss + ocr_loss

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, caption_pred_labels = torch.max(caption_pred, dim=1)
            correct_captions += (caption_pred_labels == caption_labels).sum().item()
            _, ocr_pred_labels = torch.max(ocr_pred, dim=1)
            correct_ocr += (ocr_pred_labels == ocr_labels).sum().item()
            total += caption_labels.size(0)

            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {total_loss / (batch_idx+1):.4f}, Caption Accuracy: {correct_captions / total:.4f}, OCR Accuracy: {correct_ocr / total:.4f}"
                )

        # Validate
        val_loss, val_caption_acc, val_ocr_acc = validate(
            model, test_loader, tokenizer, criterion, args.device
        )
        print(
            f"Validation Loss: {val_loss:.4f}, Caption Accuracy: {val_caption_acc:.4f}, OCR Accuracy: {val_ocr_acc:.4f}"
        )

        # Save the model backbone
        save_backbone(model, f"{args.save_dir}/visobert_backbone_epoch_{epoch+1}.pth")


# Validation function
def validate(model, test_loader, tokenizer, criterion, device):
    model.eval()
    total_val_loss = 0.0
    correct_captions = 0
    correct_ocr = 0
    total = 0

    with torch.no_grad():
        for (captions, ocr_texts), (caption_labels, ocr_labels) in test_loader:
            # Tokenize input
            caption_inputs = tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            ocr_inputs = tokenizer(
                ocr_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)

            # Prepare labels
            caption_labels = torch.tensor(caption_labels).to(device)
            ocr_labels = torch.tensor(ocr_labels).to(device)

            # Forward pass
            caption_pred, ocr_pred = model(caption_inputs, ocr_inputs)

            # Compute loss
            caption_loss = criterion(caption_pred, caption_labels)
            ocr_loss = criterion(ocr_pred, ocr_labels)
            loss = caption_loss + ocr_loss
            total_val_loss += loss.item()

            # Calculate accuracy
            _, caption_pred_labels = torch.max(caption_pred, dim=1)
            correct_captions += (caption_pred_labels == caption_labels).sum().item()
            _, ocr_pred_labels = torch.max(ocr_pred, dim=1)
            correct_ocr += (ocr_pred_labels == ocr_labels).sum().item()
            total += caption_labels.size(0)

    average_val_loss = total_val_loss / len(test_loader)
    caption_accuracy = correct_captions / total
    ocr_accuracy = correct_ocr / total
    return average_val_loss, caption_accuracy, ocr_accuracy


# Save the backbone
def save_backbone(model, save_path):
    torch.save(model.bert.state_dict(), save_path)
    print(f"Backbone saved to {save_path}")


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Sarcasm Detection Model with ViSoBERT"
    )
    parser.add_argument(
        "--metadata_path", type=str, required=True, help="Path to the metadata file"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the image directory"
    )
    parser.add_argument(
        "--train_list_id", type=str, required=True, help="Path to train list IDs"
    )
    parser.add_argument(
        "--test_list_id", type=str, required=True, help="Path to test list IDs"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--model_name", type=str, default="uitnlp/visobert", help="Model name or path"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./", help="Directory to save model"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_clipping", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=[1.0, 5.0],
        help="Class weights for loss",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")

    args = parser.parse_args()
    train_model(args)
