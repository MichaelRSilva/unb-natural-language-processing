"""
Main module for BERT-based text classification.
This script orchestrates the complete workflow including data loading,
model training, and evaluation.
"""

from load_data import load_dataset
from tokenizer import get_tokenizer, tokenize_texts
from bert_dataset import BertDataset
from train import get_model, train_model
from evaluate import evaluate_model

# Define input parameters for data loading
data_file = "data/Dmoz-Health.csv"  # Path to the input CSV file
text_column = "text"                # Name of column containing text data
label_column = "class"              # Name of column containing class labels

# Load and preprocess the dataset, splitting into train/val/test sets
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder = load_dataset(
    data_file, text_column, label_column
)

# Initialize the BERT tokenizer
tokenizer = get_tokenizer()

# Tokenize the text data for each split
# Tokenize training data and get analysis samples
train_encodings, train_sample_text, train_tokens, train_token_ids = tokenize_texts(train_texts, tokenizer)

# Tokenize validation and test data (only need encodings)
val_encodings = tokenize_texts(val_texts, tokenizer)[0]
test_encodings = tokenize_texts(test_texts, tokenizer)[0]

# Print tokenization analysis for a training sample
print("\n **Train Tokenization Analysis** ")
print(f"Sample Text: {train_sample_text}")
print(f"Tokens: {train_tokens}")
print(f"Token IDs: {train_token_ids}\n")

# Create PyTorch datasets for model training
train_dataset = BertDataset(train_encodings, train_labels)
val_dataset = BertDataset(val_encodings, val_labels)
test_dataset = BertDataset(test_encodings, test_labels)

# Load model
num_labels = len(label_encoder.classes_)
model = get_model(num_labels)

# Train model
trainer = train_model(model, train_dataset, val_dataset, tokenizer)

# Evaluate model
evaluate_model(trainer, test_dataset, test_labels, label_encoder)
