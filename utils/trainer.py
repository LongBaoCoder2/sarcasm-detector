import torch

# import torch.nn.functional as F
# from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from collections import Counter

from utils.loss import FocalLoss, get_cosine_schedule_with_warmup


num_classes = 4


def validate_model(model, test_dataloader, device="cuda"):
    model.eval()
    all_val_preds, all_val_labels = [], []
    label_counts = Counter()

    val_pbar = tqdm(test_dataloader, desc="(Validation)", leave=False)

    with torch.no_grad():
        for (images, image_texts, texts), labels, _ in val_pbar:
            images = images.to(device)
            image_texts = image_texts.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            main_output, _, _ = model(images, image_texts, texts)
            _, predicted = torch.max(main_output, 1)

            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

            label_counts.update(predicted.cpu().numpy())

    # Calculate macro-averaged metrics
    val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
    val_precision = precision_score(all_val_labels, all_val_preds, average="macro")
    val_recall = recall_score(all_val_labels, all_val_preds, average="macro")

    # Print overall metrics
    print(
        f"Overall F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
    )

    # Print classification report (metrics per label)
    print("\nClassification Report:")
    print(
        classification_report(
            all_val_labels,
            all_val_preds,
            target_names=[
                "not-sarcasm",
                "text-sarcasm",
                "image-sarcasm",
                "multi-sarcasm",
            ],
        )
    )

    # Print label counts
    print("\nPredicted Label Counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} occurrences")

    return val_f1, val_precision, val_recall, label_counts


def train_model(
    model,
    train_dataset,
    test_dataset,
    collate_fn,
    batch_size=128,
    num_epochs=30,
    learning_rate=1e-4,
    patience=7,
    device="cuda",
):
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Calculate class weights based on the train dataset
    #     train_labels = [label for _, label, _ in train_dataset]
    #     label_counts = torch.bincount(torch.tensor(train_labels))
    #     total_samples = len(train_labels)
    #     class_weights = total_samples / (len(label_counts) * label_counts.float())

    #     train_labels = np.array(train_dataset.get_all_labels())
    #     class_sample_count = np.array(
    #         [len(np.where(train_labels == t)[0]) for t in range(num_classes)])
    #     weight = 1.0 / class_sample_count
    #     weight[1:3] /= 10.
    #     samples_weight = np.array([weight[t] for t in train_labels])

    #     samples_weight = np.array([1.0, 4.0, 3.0, 1.0])
    #     samples_weight = torch.from_numpy(samples_weight)
    #     samples_weigth = samples_weight.double()
    #     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    #     label_weights = [0.1, 2.0, 1.0, 0.2]
    #     sample_weights = [label_weights[label] for label in train_labels]
    #     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    class_weights = torch.tensor([1.0, 4.0, 3.0, 1.0], dtype=torch.float32).to(device)
    class_weights = class_weights.to(device)

    # Use Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss(alpha=class_weights, gamma=2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=5e-2
    )

    # Implement warmup and cosine annealing
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.to(device)

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        all_train_preds, all_train_labels = [], []

        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} (Training)",
            leave=False,
        )

        iteration = 0
        for (images, image_texts, texts), labels, _ in train_pbar:
            images = images.to(device)
            image_texts = image_texts.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            main_output, text_aux_output, image_aux_output = model(
                images, image_texts, texts
            )

            main_loss = criterion(main_output, labels)
            text_sarcasm_mask = labels == 1
            image_sarcasm_mask = labels == 2

            text_aux_loss = (
                criterion(text_aux_output[text_sarcasm_mask], labels[text_sarcasm_mask])
                if text_sarcasm_mask.any()
                else 0
            )
            image_aux_loss = (
                criterion(
                    image_aux_output[image_sarcasm_mask], labels[image_sarcasm_mask]
                )
                if image_sarcasm_mask.any()
                else 0
            )

            loss = main_loss + 0.2 * (text_aux_loss + image_aux_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(main_output, 1)

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            iteration += 1

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_loss / len(train_dataloader)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
        train_precision = precision_score(
            all_train_labels, all_train_preds, average="macro"
        )
        train_recall = recall_score(all_train_labels, all_train_preds, average="macro")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        )

        # Validation phase
        val_f1, val_precision, val_recall, label_counts = validate_model(
            model, test_dataloader
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_backbone.pth")
            print(f"Best model backbone saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete.")
