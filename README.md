# sarcasm-detector

This project implements a multimodal__ sarcasm__ detection model that combines image and text features to classify different types of sarcasm. The model is designed to effectively fuse features from multiple modalities, handle imbalanced datasets, and improve generalization using auxiliary tasks and specialized loss functions.

## Features of the Architecture

1. __Low-Rank Fusion Module__: Compresses and combines features from the text and image embeddings to reduce dimensionality while maintaining essential information.
2. __Text Encoder__: The model uses a pre-trained text encoder __ViSoBERT__
3. __Image Encoder__: A CLIP-based image encoder extracts visual features from input images.

## Overcoming Imbalanced Datasets
1. __Focal Loss__: This loss function gives more focus to minority classes (e.g., text-sarcasm and image-sarcasm) by dynamically scaling the gradient of easy samples.
2. Auxiliary Losses:
    - For text-sarcasm: The model ensures text-based features dominate the prediction by leveraging text-specific auxiliary outputs.
    - For image-sarcasm: Fused embeddings are emphasized for prediction, ensuring better understanding of visual sarcasm.
