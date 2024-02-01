from transformers import DistilBertModel
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, dropout_rate=0.1, extra_layers=0):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.extra_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(extra_layers)])

    def forward(self, x):
        x = self.pre_classifier(x)
        x = nn.ReLU()(x)

        for layer in self.extra_layers:
            x = layer(x)
            x = nn.ReLU()(x)

        x = self.dropout(x)
        x = self.classifier(x)

        return x


class CustomDistilBERT(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.1, extra_layers=0):
        super(CustomDistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert.eval()  # Set DistilBERT to eval mode
        self.classifier = Classifier(self.distilbert.config.dim, self.distilbert.config.dim, num_labels, dropout_rate, extra_layers)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output.last_hidden_state[:, 0]
        
        return self.classifier(hidden_state)
