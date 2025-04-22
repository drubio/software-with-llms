import html
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse, re, os, json
import requests
from collections import Counter
from pathlib import Path

# ---------- Configuration ----------
MODEL_CACHE_DIR = "lstm_models"
VOCAB_CACHE_DIR = "lstm_vocab"

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
    vocab['<pad>'] = 0
    vocab['<unk>'] = len(vocab)  # Last index
    
    print(f"[INFO] Built vocabulary with {len(vocab)} words")
    return vocab

def encode(tokens, vocab):
    # Encode tokens using vocabulary, with unknown token handling
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, data, context=3):
        self.data = data
        self.context = context

    def __len__(self):
        return len(self.data) - self.context

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.context]),
            torch.tensor(self.data[idx+self.context]),
        )

# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.linear(h_n[-1])

# ---------- Training ----------
def train_model(model, loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            
        avg_loss = total_loss / total_batches
        print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model

# ---------- Prediction ----------
def predict_next(model, vocab, idx_to_word, prompt, context, top_k=5, temperature=1.0, max_words=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    tokens = basic_tokenize(prompt)[-context:]
    generated = tokens.copy()

    for _ in range(max_words):
        if len(generated) < context:
            # Pad with start tokens if needed
            padding = ['<pad>'] * (context - len(generated))
            context_tokens = padding + generated
        else:
            context_tokens = generated[-context:]
        
        # Convert tokens to tensor
        input_ids = torch.tensor([
            [vocab.get(tok, vocab['<unk>']) for tok in context_tokens]
        ], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_ids)

                        # Handle temperature differently
            if temperature == 0:
                # For temperature=0, just use argmax (no randomness)
                top_indices = torch.topk(output, min(top_k, output.size(-1))).indices.squeeze()
                # Set probability to 1.0 for the top item for display purposes
                top_probs = [1.0] + [0.0] * (min(top_k, output.size(-1)) - 1)
            else:
                # Normal case with temperature > 0
                probs = torch.softmax(output / temperature, dim=-1).squeeze()
                topk_results = torch.topk(probs, min(top_k, len(probs)))
                top_indices = topk_results.indices.tolist()
                top_probs = topk_results.values.tolist()

            # Convert indices to words
            if isinstance(top_indices, torch.Tensor):
                # Handle the case where top_indices is a tensor (temperature=0 case)
                options = [(idx_to_word.get(i.item(), '<unk>'), p) 
                           for i, p in zip(top_indices, top_probs)]
            else:
                # Handle the case where top_indices is already a list
                options = [(idx_to_word.get(i, '<unk>'), p) 
                           for i, p in zip(top_indices, top_probs)]

        print("[DEBUG] Top predictions:")
        for word, prob in options:
            print(f"  {word}: {prob:.4f}")

        # Add the top prediction to the generated text
        generated.append(options[0][0])

    # Return everything after the original prompt
    return " ".join(generated[len(tokens):])

# ---------- File Management ----------
def get_cache_paths(args):
    # Create necessary directories
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    os.makedirs(VOCAB_CACHE_DIR, exist_ok=True)
    
    # Generate filenames with clear suffixes
    corpus_type = "nursery" if args.nursery else "shakespeare"
    vocab_file = Path(VOCAB_CACHE_DIR) / f"vocab_{corpus_type}_ctx{args.context}.json"
    model_file = Path(MODEL_CACHE_DIR) / f"model_{corpus_type}_ctx{args.context}.pt"
    
    return vocab_file, model_file

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="LSTM Language Model")
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--nursery", action="store_true", help="Use nursery rhyme corpus")
    parser.add_argument("--context", type=int, default=3, help="Context window size")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--maxwords", type=int, default=1, help="Max number of predicted words")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if model exists")
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
    dataset = TextDataset(encoded, context=args.context)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model with the correct vocabulary size
    vocab_size = len(vocab)  # Number of unique words including special tokens
    model = LSTMModel(vocab_size)

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
    print(f"LSTM Prediction for '{args.prompt}': {prediction}")

if __name__ == "__main__":
    main()
