"""
This module handles the evaluation of trained BERT models.
It provides functionality for computing various performance metrics 
including accuracy, F1-scores, and confusion matrix visualization.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(trainer, test_dataset, test_labels, label_encoder):
    """
    Evaluate the trained model on the test dataset using multiple metrics.
    
    Args:
        trainer (Trainer): The trained HuggingFace Trainer instance
        test_dataset (BertDataset): Dataset for testing
        test_labels (pd.Series): True labels for the test set
        label_encoder (LabelEncoder): Encoder used to transform labels
        
    Returns:
        tuple: A tuple containing (accuracy, f1_micro, f1_macro, confusion_matrix)
    """
    # Get model predictions
    y_pred = trainer.predict(test_dataset).predictions
    y_pred = np.argmax(y_pred, axis=1)  # Convert logits to class predictions
    y_true = test_labels.to_numpy()      # Convert labels to numpy array

    # Calculate multiple evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)               # Overall accuracy
    f1_micro = f1_score(y_true, y_pred, average='micro')   # Micro-averaged F1 score
    f1_macro = f1_score(y_true, y_pred, average='macro')   # Macro-averaged F1 score
    conf_matrix = confusion_matrix(y_true, y_pred)         # Confusion matrix

    return accuracy, f1_micro, f1_macro, conf_matrix
