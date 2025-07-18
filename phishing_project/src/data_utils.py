import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import numpy as np

class PhishDataset(Dataset):
    """Dataset class for phishing URL detection."""
    
    def __init__(self, urls, labels, tokenizer, max_length=128):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = str(self.urls[idx])
        label = self.labels[idx] if self.labels is not None else 0
        
        # Tokenize the URL
        encoding = self.tokenizer(
            url,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(train_path, test_path=None, test_size=0.2, random_state=42):
    """
    Load and preprocess the phishing detection data.
    
    Args:
        train_path (str): Path to training CSV file
        test_path (str): Path to test CSV file (optional)
        test_size (float): Fraction of train data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Assuming columns are 'url' and 'label' (adjust as needed)
    if 'url' not in train_df.columns:
        # If columns are different, assume first column is URL, second is label
        train_df.columns = ['url', 'label']
    
    # Stratified split to preserve class balance
    X = train_df['url']
    y = train_df['label']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_split = pd.DataFrame({'url': X_train, 'label': y_train})
    val_split = pd.DataFrame({'url': X_val, 'label': y_val})
    
    # Load test data if provided
    test_df = None
    if test_path:
        test_df = pd.read_csv(test_path)
        if 'url' not in test_df.columns:
            # Assume first column is URL
            test_df.columns = ['url'] if len(test_df.columns) == 1 else ['url'] + [f'col_{i}' for i in range(1, len(test_df.columns))]
    
    return train_split, val_split, test_df

def create_data_loaders(train_df, val_df, tokenizer, batch_size=16, max_length=128):
    """
    Create DataLoader objects for training and validation.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        tokenizer: DistilBERT tokenizer
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = PhishDataset(
        train_df['url'].values,
        train_df['label'].values,
        tokenizer,
        max_length
    )
    
    val_dataset = PhishDataset(
        val_df['url'].values,
        val_df['label'].values,
        tokenizer,
        max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for CPU compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

def create_test_loader(test_df, tokenizer, batch_size=16, max_length=128):
    """Create DataLoader for test predictions."""
    test_dataset = PhishDataset(
        test_df['url'].values,
        None,  # No labels for test data
        tokenizer,
        max_length
    )
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )