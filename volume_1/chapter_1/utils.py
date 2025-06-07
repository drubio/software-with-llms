"""
Shared utility functions for all language model implementations.
"""
import html
import re
import os
import json
import requests
import unicodedata
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration
CACHE_DIR = "model_cache"
CORPUS_CACHE_DIR = "corpus_cache"
VOCAB_CACHE_DIR = "vocab_cache"

# Unified Text Processing
def basic_tokenize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace('-', ' ')
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.strip().split()
    cleaned_tokens = []
    for token in tokens:
        token = unicodedata.normalize("NFKC", token).strip()
        if re.match(r'^[a-z]+\d+$', token):
            token = re.sub(r'\d+$', '', token)
        cleaned_tokens.append(token)
    return cleaned_tokens

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
            urls = [
                "https://www.gutenberg.org/files/39784/39784-h/39784-h.htm",
                "https://www.poetryfoundation.org/poems/43200/twinkle-twinkle-little-star"
            ]
            full_text = ""
            for url in urls:
                print(f"[INFO] Fetching: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator=' ', strip=True)
                start = text.find("Mother Goose")
                end = text.find("End of the Project Gutenberg")
                if start != -1 and end != -1:
                    text = text[start:end]
                lines = text.splitlines()
                cleaned_lines = [
                    line for line in lines
                    if 'project gutenberg' not in line.lower()
                    and 'musicxml' not in line.lower()
                    and 'http' not in line.lower()
                    and len(line.strip()) > 0
                ]
                full_text += ' ' + ' '.join(cleaned_lines)
            text = full_text.strip()
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
        special_tokens = ['<pad>', '<unk>', '<bos>']

    # Count word frequencies
    word_counts = Counter(tokens)
    vocab = {}
    idx = 1
    vocab['<pad>'] = 0
    for word, count in word_counts.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    print(f"[INFO] Built vocabulary with {len(vocab)} words")
    return vocab

def encode(tokens, vocab):
    """Encode tokens using vocabulary, with unknown token handling."""
    return [vocab.get(token, vocab.get('<unk>', 0)) for token in tokens]

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
