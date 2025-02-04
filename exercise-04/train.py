import torch
import numpy as np
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

def get_model(num_labels):
    """
    Initialize a BERT model for sequence classification.
    
    Args:
        num_labels (int): Number of unique labels in the classification task
        
    Returns:
        BertForSequenceClassification: Initialized BERT model
    """
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        eval_pred (tuple): Contains (predictions, labels)
            - predictions: Model output logits
            - labels: True labels
            
    Returns:
        dict: Dictionary containing accuracy metric
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def train_model(model, train_dataset, val_dataset, tokenizer):
    """
    Train the BERT model using the provided datasets.
    
    Args:
        model (BertForSequenceClassification): The initialized BERT model
        train_dataset (BertDataset): Dataset for training
        val_dataset (BertDataset): Dataset for validation
        tokenizer (BertTokenizer): BERT tokenizer instance
        
    Returns:
        Trainer: Trained model trainer instance
    """
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Directory for storing model checkpoints
        evaluation_strategy='epoch',     # Evaluate after each epoch
        save_strategy='epoch',          # Save model after each epoch
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,   # Batch size for evaluation
        num_train_epochs=3,             # Number of training epochs
        weight_decay=0.01,              # Weight decay for regularization
        logging_dir='./logs',           # Directory for storing logs
        logging_steps=10,               # Log every 10 steps
        load_best_model_at_end=True,    # Load the best model when finished
        metric_for_best_model="accuracy" # Use accuracy to determine best model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Pass accuracy computation
    )

    trainer.train()
    return trainer
