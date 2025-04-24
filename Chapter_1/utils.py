"""
Shared utility functions for all language model implementations.
"""
import html
import re
import os
import json
import requests
from collections import Counter
from pathlib import Path

# Configuration
CACHE_DIR = "model_cache"
CORPUS_CACHE_DIR = "corpus_cache"
VOCAB_CACHE_DIR = "vocab_cache"

# Text processing
def basic_tokenize(text):
    """Normalize and tokenize text into words."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    return text.lower().split()

# Corpus loading
def load_corpus(corpus_type, use_cache=True):
    """Load corpus from source or cache."""
    # Create cache directory
    os.makedirs(CORPUS_CACHE_DIR, exist_ok=True)
    
    cache_file = Path(CORPUS_CACHE_DIR) / f"{corpus_type}_corpus.txt"
    
    # Return from cache if available and requested
    if use_cache and cache_file.exists():
        print(f"[INFO] Loading corpus from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    
    # Otherwise download fresh
    print(f"[INFO] Downloading {corpus_type} corpus...")
    
    try:
        if corpus_type == "nursery":
            url = "https://www.gutenberg.org/files/38562/38562-h/38562-h.htm"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            body = re.findall(r"<body.*?>(.*?)</body>", html_content, re.DOTALL)[0]
            text = re.sub(r"<.*?>", "", body)
            text = html.unescape(text)
        elif corpus_type == "shakespeare":
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")
        
        # Cache the corpus
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(text)
            
        return text
    except Exception as e:
        print(f"[ERROR] Failed to download corpus: {e}")
        # Fallback to cache if exists
        if cache_file.exists():
            print(f"[INFO] Falling back to cached corpus")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        raise

# Vocabulary handling
def build_vocab(tokens, min_freq=2, special_tokens=None):
    """Build vocabulary from tokens with optional special tokens."""
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>']
    
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Filter by frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create vocabulary (starting from 1, 0 reserved for padding)
    vocab = {word: i + 1 for i, word in enumerate(sorted(filtered_words))}
    
    # Add special tokens
    next_idx = len(vocab) + 1  # +1 because we started indexing at 1
    for token in special_tokens:
        if token == '<pad>':
            vocab[token] = 0  # Padding is always 0
        else:
            vocab[token] = next_idx
            next_idx += 1
    
    print(f"[INFO] Built vocabulary with {len(vocab)} words")
    return vocab

def encode(tokens, vocab):
    """Encode tokens using vocabulary, with unknown token handling."""
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def save_vocab(vocab, vocab_file):
    """Save vocabulary to file."""
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)
    print(f"[INFO] Saved vocabulary to {vocab_file}")

def load_vocab(vocab_file):
    """Load vocabulary from file."""
    with open(vocab_file, "r") as f:
        vocab_data = json.load(f)
        vocab = {k: int(v) for k, v in vocab_data.items()}
    print(f"[INFO] Loaded vocabulary from {vocab_file}")
    return vocab
