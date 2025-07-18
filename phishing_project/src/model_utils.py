import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup

class PhishingClassifier(nn.Module):
    """DistilBERT-based phishing URL classifier."""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super(PhishingClassifier, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding tokens
        
        Returns:
            logits: Raw predictions for each class
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def setup_model_and_tokenizer(model_name='distilbert-base-uncased'):
    """
    Initialize model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model
    
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = PhishingClassifier(model_name)
    
    return model, tokenizer

def setup_optimizer_and_scheduler(model, train_loader, epochs=3, lr=2e-5, weight_decay=0.01):
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        train_loader: Training data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler