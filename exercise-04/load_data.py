"""
This module handles data loading and preprocessing for the BERT text classification model.
It includes functionality for loading CSV data, splitting into train/val/test sets,
and encoding labels.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path, text_column, label_column):
    """
    Load and preprocess the dataset for BERT text classification.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
        text_column (str): Name of the column containing the text data
        label_column (str): Name of the column containing the labels
        
    Returns:
        tuple: A tuple containing:
            - train_texts (pd.Series): Text data for training
            - val_texts (pd.Series): Text data for validation
            - test_texts (pd.Series): Text data for testing
            - train_labels (pd.Series): Labels for training
            - val_labels (pd.Series): Labels for validation
            - test_labels (pd.Series): Labels for testing
            - label_encoder (LabelEncoder): Fitted label encoder for categorical labels
            
    Raises:
        FileNotFoundError: If the specified file_path does not exist
        ValueError: If the specified columns are not found in the dataset
    """
    df = pd.read_csv(file_path)

    # Ensure the required columns exist
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"Columns '{text_column}' and '{label_column}' must exist in the dataset.")

    df = df[[text_column, label_column]].rename(columns={text_column: 'text', label_column: 'label'})

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.67, random_state=42
    )

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder
