import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

from data_utils import load_and_preprocess_data, create_data_loaders
from model_utils import setup_model_and_tokenizer, setup_optimizer_and_scheduler

# Configuration
CONFIG = {
    'train_path': 'data/Train.csv',
    'test_path': 'data/Test.csv',
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'random_state': 42,
    'early_stopping_patience': 2,
    'output_dir': 'outputs/'
}

def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probabilities)
    
    return avg_loss, accuracy, f1, auc

def main():
    # Set random seed for reproducibility
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_preprocess_data(
        CONFIG['train_path'],
        CONFIG['test_path'],
        random_state=CONFIG['random_state']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df) if test_df is not None else 0}")
    
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(CONFIG['model_name'])
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, tokenizer,
        batch_size=CONFIG['batch_size'],
        max_length=CONFIG['max_length']
    )
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model, train_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Training loop
    print("Starting training...")
    best_auc = 0
    patience_counter = 0
    training_history = []
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate
        val_loss, accuracy, f1, auc = evaluate_model(model, val_loader, device)
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
        training_history.append(metrics)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        # Save best model based on AUC
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            
            # Save model state and tokenizer separately to avoid serialization issues
            model_save_path = os.path.join(CONFIG['output_dir'], 'model.pt')
            tokenizer_save_path = os.path.join(CONFIG['output_dir'], 'tokenizer')
            
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG,
                'metrics': metrics
            }, model_save_path)
            
            # Save tokenizer separately
            tokenizer.save_pretrained(tokenizer_save_path)
            
            print(f"New best model saved! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(CONFIG['output_dir'], 'training_history.csv'), index=False)
    
    print(f"\nTraining completed! Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {os.path.join(CONFIG['output_dir'], 'model.pt')}")

if __name__ == "__main__":
    main()