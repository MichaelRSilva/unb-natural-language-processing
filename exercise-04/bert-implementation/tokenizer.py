"""
This module provides tokenization utilities for BERT-based text processing.
It includes functions for initializing a BERT tokenizer and processing text data.
"""

import pandas as pd
from transformers import BertTokenizer


def get_tokenizer():
    """
    Initialize and return a pre-trained BERT tokenizer.
    
    Returns:
        BertTokenizer: A BERT tokenizer instance initialized with the 'bert-base-uncased' model
    """
    return BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, tokenizer):
    """
    Tokenize input texts using the provided BERT tokenizer.
    
    Args:
        texts (Union[str, List[str], pd.Series]): Input text(s) to tokenize. Can be a single string,
            a list of strings, or a pandas Series containing strings
        tokenizer (BertTokenizer): The BERT tokenizer instance to use for tokenization
    
    Returns:
        tuple: A tuple containing:
            - encodings: The tokenized and encoded text
            - sample_text: A sample of the original text (first item)
            - sample_tokens: Tokens for the sample text
            - sample_token_ids: Token IDs for the sample text
            
    Raises:
        ValueError: If the input texts are not in the correct format
    """
    # Convert single string to list for consistent processing
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to a list

    # Handle pandas Series input
    if isinstance(texts, pd.Series):
        texts = texts.tolist()  # Convert Pandas Series to a list of strings

    # Validate input format
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Expected input to be a string, Pandas Series, or a list of strings.")

    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Token Analysis Prints (for the first sample)
    sample_text = texts[0] if len(texts) > 0 else "No text available"
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return encodings, sample_text, tokens, token_ids
