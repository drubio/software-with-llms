import html
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse, re, os, json, math
import requests
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ---------- Configuration ----------
MODEL_CACHE_DIR = "transformer_models"
VOCAB_CACHE_DIR = "transformer_vocab"

# ---------- Corpus Loading ----------
def basic_tokenize(text):
    # Normalize whitespace (convert multiple spaces, tabs, newlines to 1 space)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    return text.lower().split()

def load_corpus(use_nursery):
    if use_nursery:
        print("[INFO] Loading nursery rhyme corpus...")
        url = "https://www.gutenberg.org/files/38562/38562-h/38562-h.htm"
        html_content = requests.get(url).text
        body = re.findall(r"<body.*?>(.*?)</body>", html_content, re.DOTALL)[0]
        text = re.sub(r"<.*?>", "", body)
        text = html.unescape(text)                
    else:
        print("[INFO] Loading Tiny Shakespeare corpus...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
    return basic_tokenize(text)

# ---------- Vocab & Encoding ----------
def build_vocab(tokens, min_freq=2):
    # Use Counter to count word frequencies
    word_counts = Counter(tokens)
    
    # Filter words by frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create vocabulary with indices starting from 1 (0 reserved for padding)
    vocab = {word: i + 1 for i, word in enumerate(sorted(filtered_words))}
    
    # Add special tokens
    vocab['<pad>'] = 0  # Padding token
    vocab['<unk>'] = len(vocab)  # Unknown token
    vocab['<cls>'] = len(vocab)  # Classification token (used for next token prediction)
    vocab['<sep>'] = len(vocab)  # Separator token
    
    print(f"[INFO] Built vocabulary with {len(vocab)} words")
    return vocab

def encode(tokens, vocab):
    # Encode tokens using vocabulary, with unknown token handling
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# ---------- Dataset ----------
class TransformerDataset(Dataset):
    def __init__(self, data, context=8):
        self.data = data
        self.context = context

    def __len__(self):
        return len(self.data) - self.context

    def __getitem__(self, idx):
        # Input sequence is context tokens
        input_seq = self.data[idx:idx+self.context]
        # Target is the next token after the context
        target = self.data[idx+self.context]
        
        return torch.tensor(input_seq), torch.tensor(target)

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Initialize encoding buffer
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (won't be updated during backprop)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1), :]

# ---------- Transformer Model ----------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=512, dropout=0.1, max_len=5000):
        super().__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Stack of transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output linear layer (prediction)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_weights()
        
        # Model dimension
        self.d_model = d_model
        
    def _init_weights(self):
        # Initialize embedding weights
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding token embedding to zeros
        self.embedding.weight.data[0].zero_()
        
    def forward(self, src, src_mask=None):
        # Create source mask (hide padding tokens)
        if src_mask is None:
            src_mask = src == 0
        
        # Embed tokens and apply positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # Get prediction for the last token in sequence
        output = self.output(output[:, -1, :])
        
        return output

# ---------- Training ----------
def train_model(model, loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Get model predictions
            output = model(x)
            
            # Calculate loss
            loss = criterion(output, y)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Update parameters
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / total_batches
        print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
    
    return model

# ---------- Prediction ----------
def predict_next(model, vocab, idx_to_word, prompt, context, top_k=5, temperature=1.0, max_words=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Tokenize input prompt
    tokens = basic_tokenize(prompt)
    generated = tokens.copy()

    # Generate specified number of words
    for _ in range(max_words):
        # Get the last 'context' tokens
        if len(generated) < context:
            # Pad with zeros if needed
            context_tokens = generated.copy()
            while len(context_tokens) < context:
                context_tokens.insert(0, '<pad>')
        else:
            context_tokens = generated[-context:]
        
        # Convert tokens to tensor
        input_ids = torch.tensor([
            [vocab.get(tok, vocab['<unk>']) for tok in context_tokens]
        ], dtype=torch.long).to(device)

        # Generate prediction
        with torch.no_grad():
            output = model(input_ids)

            # Apply temperature and get top-k predictions
            if temperature == 0:
                # For temperature=0, just use argmax (no randomness)
                top_indices = torch.topk(output, min(top_k, output.size(-1))).indices.squeeze()
                # Set probability to 1.0 for the top item
                top_probs = [1.0] + [0.0] * (min(top_k, output.size(-1)) - 1)
                next_token_id = top_indices[0].item()
            else:
                # Apply temperature scaling
                scaled_logits = output / temperature
                probs = torch.softmax(scaled_logits, dim=-1).squeeze()
                
                # Get top-k predictions
                topk_results = torch.topk(probs, min(top_k, len(probs)))
                top_indices = topk_results.indices.tolist()
                top_probs = topk_results.values.tolist()
                
                # Sample from the distribution
                if isinstance(top_indices, int):
                    # Handle single element case
                    next_token_id = top_indices
                    top_indices = [top_indices]
                    top_probs = [top_probs]
                else:
                    # Sample from top-k tokens according to probabilities
                    next_token_id = np.random.choice(top_indices, p=np.array(top_probs)/sum(top_probs))

            # Convert indices to words
            options = [(idx_to_word.get(i, '<unk>'), p) 
                      for i, p in zip(top_indices, top_probs)]

        # Print top predictions
        print("[DEBUG] Top predictions:")
        for word, prob in options:
            print(f"  {word}: {prob:.4f}")

        # Add the next token to generated text
        next_word = idx_to_word.get(next_token_id, '<unk>')
        generated.append(next_word)

    # Return everything after the original prompt
    return " ".join(generated[len(tokens):])

# ---------- File Management ----------
def get_cache_paths(args):
    # Create necessary directories
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(VOCAB_CACHE_DIR, exist_ok=True)
    
    # Generate filenames
    corpus_type = "nursery" if args.nursery else "shakespeare"
    vocab_file = Path(VOCAB_CACHE_DIR) / f"vocab_{corpus_type}_ctx{args.context}.json"
    model_file = Path(MODEL_CACHE_DIR) / f"model_{corpus_type}_ctx{args.context}.pt"
    
    return vocab_file, model_file

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Transformer Language Model")
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--nursery", action="store_true", help="Use nursery rhyme corpus")
    parser.add_argument("--context", type=int, default=8, help="Context window size")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--maxwords", type=int, default=5, help="Max number of predicted words")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if model exists")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    args = parser.parse_args()

    # Get appropriate cache paths based on arguments
    vocab_file, model_file = get_cache_paths(args)

    # Load the corpus
    tokens = load_corpus(args.nursery)
    print(f"[INFO] Loaded corpus with {len(tokens)} tokens")

    # Build or load vocabulary
    if vocab_file.exists() and not args.force_train:
        with open(vocab_file, "r") as f:
            vocab_data = json.load(f)
            vocab = {k: int(v) for k, v in vocab_data.items()}
        print(f"[INFO] Loaded vocabulary from {vocab_file}")
    else:
        vocab = build_vocab(tokens, min_freq=2)
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)
        print(f"[INFO] Saved vocabulary to {vocab_file}")

    # Create mapping from indices to words
    idx_to_word = {i: w for w, i in vocab.items()}
    
    # Encode the entire corpus
    encoded = encode(tokens, vocab)

    # Create dataset and dataloader
    dataset = TransformerDataset(encoded, context=args.context)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    model = TransformerModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )
    
    # Load or train the model
    if model_file.exists() and not args.force_train:
        try:
            print(f"[INFO] Loading cached model from {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        except Exception as e:
            print(f"[WARNING] Error loading model: {e}. Will train a new model.")
            print(f"[INFO] Training model with {args.epochs} epochs...")
            model = train_model(model, loader, epochs=args.epochs)
            torch.save(model.state_dict(), model_file)
            print(f"[INFO] Saved model to {model_file}")
    else:
        print(f"[INFO] Training model with {args.epochs} epochs...")
        model = train_model(model, loader, epochs=args.epochs)
        torch.save(model.state_dict(), model_file)
        print(f"[INFO] Saved model to {model_file}")

    # Generate prediction
    prediction = predict_next(
        model, vocab, idx_to_word,
        args.prompt,
        context=args.context,
        top_k=args.topk,
        temperature=args.temperature,
        max_words=args.maxwords
    )
    print(f"Transformer Prediction for '{args.prompt}': {prediction}")

if __name__ == "__main__":
    main()
