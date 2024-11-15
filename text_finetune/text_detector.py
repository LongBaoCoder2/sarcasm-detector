import torch.nn as nn


class SarcasmClassifier(nn.Module):
    def __init__(self, text_embeder, dropout=0.3):
        super(SarcasmClassifier, self).__init__()
        self.bert = text_embeder  # ViSoBERT
        self.dropout = nn.Dropout(dropout)
        # Two separate classifiers for caption and OCR text
        self.caption_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.ocr_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, caption_input, ocr_input):
        # Forward pass through BERT
        caption_output = self.bert(**caption_input).last_hidden_state[:, 0, :]
        #         caption_output = mean_pooling(caption_output, caption_input['attention_mask'])

        ocr_output = self.bert(**ocr_input).last_hidden_state[:, 0, :]
        #         ocr_output = mean_pooling(ocr_output, ocr_input['attention_mask'])

        # Classifier heads
        caption_pred = self.caption_classifier(self.dropout(caption_output))
        ocr_pred = self.ocr_classifier(self.dropout(ocr_output))

        return caption_pred, ocr_pred
