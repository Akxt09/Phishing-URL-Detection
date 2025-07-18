import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from data_utils import create_test_loader
from model_utils import PhishingClassifier

from transformers import DistilBertTokenizer

def load_trained_model(model_path, device):
    """Load the trained model and tokenizer."""
    # Fix for PyTorch 2.6+ weights_only default change
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load tokenizer from saved directory
    tokenizer_path = model_path.replace('model.pt', 'tokenizer')
    if os.path.exists(tokenizer_path):
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    else:
        # Fallback to default tokenizer if separate save doesn't exist
        tokenizer = DistilBertTokenizer.from_pretrained(checkpoint['config']['model_name'])
    
    # Recreate model
    model = PhishingClassifier(checkpoint['config']['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, checkpoint['config']

def generate_predictions(model, test_loader, device):
    """Generate predictions for test data."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Positive class probability
    
    return all_predictions, all_probabilities

def main():
    # Configuration - Use raw strings or forward slashes for Windows paths
    model_path = 'outputs/model.pt'  # Update with your model path
    test_data_path = 'data/Test.csv'  # Update with your test data path
    output_path = 'outputs/test_predictions.csv'  # Update with your output path

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first to train the model.")
        return
    
    # Load trained model
    print("Loading trained model...")
    model, tokenizer, config = load_trained_model(model_path, device)
    print("Model loaded successfully!")
    
    # Load test data
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        return
    
    test_df = pd.read_csv(test_data_path)
    if 'url' not in test_df.columns:
        # Assume first column is URL
        test_df.columns = ['url'] + [f'col_{i}' for i in range(1, len(test_df.columns))]
    
    print(f"Test samples: {len(test_df)}")
    
    # Create test data loader
    test_loader = create_test_loader(
        test_df, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    # Generate predictions
    print("Generating predictions...")
    predictions, probabilities = generate_predictions(model, test_loader, device)
    
    # Create basic results DataFrame with URLs and predictions
    basic_results_df = pd.DataFrame({
        'url': test_df['url'],
        'prediction': predictions
    })
    
    # Create detailed results DataFrame
    detailed_data = {
        'url': test_df['url'],
        'predicted_probability': probabilities,
        'predicted_label': predictions
    }
    
    # Check if test data has true labels (for evaluation purposes)
    if 'label' in test_df.columns or 'true_label' in test_df.columns:
        label_col = 'label' if 'label' in test_df.columns else 'true_label'
        detailed_data['true_label'] = test_df[label_col]
    
    detailed_results_df = pd.DataFrame(detailed_data)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    basic_results_df.to_csv(output_path, index=False)
    
    # Save detailed results with probabilities
    detailed_output_path = output_path.replace('.csv', '_detailed.csv')
    detailed_results_df.to_csv(detailed_output_path, index=False)
    
    print(f"Basic predictions saved to: {output_path}")
    print(f"Detailed predictions saved to: {detailed_output_path}")
    
    # Print summary
    phishing_count = sum(predictions)
    total_count = len(predictions)
    print(f"\nPrediction Summary:")
    print(f"Total URLs: {total_count}")
    print(f"Predicted Phishing: {phishing_count} ({phishing_count/total_count:.2%})")
    print(f"Predicted Legitimate: {total_count - phishing_count} ({(total_count - phishing_count)/total_count:.2%})")
    
    # If true labels are available, show accuracy
    if 'true_label' in detailed_results_df.columns:
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(detailed_results_df['true_label'], predictions)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(detailed_results_df['true_label'], predictions, 
                                  target_names=['Legitimate', 'Phishing']))

if __name__ == "__main__":
    main()