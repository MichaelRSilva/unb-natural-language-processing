"""
This module provides a PyTorch Dataset implementation for BERT model training.
It handles the organization and access of tokenized text data and their corresponding labels.
"""

import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    """
    A custom Dataset class for BERT model training.
    
    This class implements the PyTorch Dataset interface to efficiently
    handle batched data loading for BERT model training.
    
    Attributes:
        encodings (dict): The tokenized and encoded text data
        labels (list): The corresponding labels for each text sample
    """
    
    def __init__(self, encodings, labels):
        """
        Initialize the BertDataset.
        
        Args:
            encodings (dict): Tokenized and encoded text from the BERT tokenizer
            labels (list): Corresponding labels for the text samples
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: The number of samples
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a specific item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_ids: Tensor of token ids
                - attention_mask: Tensor of attention mask
                - labels: Tensor of the label
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item
