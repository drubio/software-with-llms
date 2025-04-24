"""
Transformer-based language model implementation.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import heapq

import utils

# ---------- Configuration ----------
MODEL_CACHE_DIR = "transformer_models"


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


def train_model(model, loader, epochs=5, device=None):
    """Train the transformer model."""
    if device is None:
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


def predict_next_beam_search(model, vocab, idx_to_word, prompt, context, beam_width=5, temperature=1.0, max_words=5):
    """Generate text predictions using beam search with the transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Tokenize input prompt
    tokens = utils.basic_tokenize(prompt)
    initial_tokens = tokens.copy()
    
    # Initial context
    if len(tokens) < context:
        # Pad with zeros if needed
        context_tokens = tokens.copy()
        while len(context_tokens) < context:
            context_tokens.insert(0, '<pad>')
    else:
        context_tokens = tokens[-context:]
    
    # Initial beam
    beams = [(0.0, context_tokens, [])]  # (score, context, generated)
    
    # Store all predictions for each step
    all_predictions = []
    
    # Generate specified number of words
    for step in range(max_words):
        candidates = []
        step_predictions = []
        
        # Process all current beams
        for score, beam_context, beam_generated in beams:
            # Convert tokens to tensor
            input_ids = torch.tensor([
                [vocab.get(tok, vocab['<unk>']) for tok in beam_context]
            ], dtype=torch.long).to(device)
            
            # Generate prediction
            with torch.no_grad():
                output = model(input_ids)
                
                # Apply temperature scaling
                if temperature > 0:
                    scaled_logits = output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1).squeeze()
                else:
                    # For temperature=0, just use the logits directly
                    probs = torch.zeros_like(output).squeeze()
                    probs[output.argmax()] = 1.0
                
                # Get top-k candidates for this beam
                topk_probs, topk_indices = torch.topk(probs, beam_width)
                
                # Store predictions for the first beam in each step
                if len(beam_generated) == step:
                    for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist()):
                        next_word = idx_to_word.get(idx, '<unk>')
                        step_predictions.append((next_word, prob))
                
                for i, (prob, idx) in enumerate(zip(topk_probs.tolist(), topk_indices.tolist())):
                    # Convert index to word
                    next_word = idx_to_word.get(idx, '<unk>')
                    
                    # Calculate new score: log probability
                    # We use log probabilities to avoid underflow with small numbers
                    new_score = score + math.log(prob + 1e-10)  # Add small epsilon to avoid log(0)
                    
                    # Create new context by shifting window
                    new_context = beam_context[1:] + [next_word]
                    
                    # Add to current generated sequence
                    new_generated = beam_generated + [next_word]
                    
                    # Add to candidates
                    candidates.append((new_score, new_context, new_generated))
        
        # Keep only the top beam_width candidates
        beams = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        
        # Add predictions for this step
        all_predictions.append(step_predictions)
    
    # Return the best beam's generated text and all predictions
    best_beam = max(beams, key=lambda x: x[0])
    generated_text = " ".join(best_beam[2])
    
    return generated_text, all_predictions, beams


def predict_next(model, vocab, idx_to_word, prompt, context, top_k=5, temperature=1.0, max_words=5, beam_width=0):
    """Generate text predictions using the transformer model."""
    # If beam search is enabled, use beam search generation
    if beam_width > 0:
        return predict_next_beam_search(
            model, vocab, idx_to_word, prompt, context, 
            beam_width=beam_width, temperature=temperature, max_words=max_words
        )
    
    # Otherwise, use regular greedy/sampling generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Tokenize input prompt
    tokens = utils.basic_tokenize(prompt)
    generated = tokens.copy()
    
    # Store predictions for each step
    all_predictions = []

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
            
            # Store predictions for this step
            all_predictions.append(options)

        # Add the next token to generated text
        next_word = idx_to_word.get(next_token_id, '<unk>')
        generated.append(next_word)

    # Return everything after the original prompt and all predictions
    return " ".join(generated[len(tokens):]), all_predictions


def get_cache_paths(corpus_type, context_size):
    """Generate appropriate cache paths for model files."""
    # Create necessary directories
    os.makedirs(utils.VOCAB_CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # Generate filenames with clear suffixes
    vocab_file = Path(utils.VOCAB_CACHE_DIR) / f"vocab_{corpus_type}_ctx{context_size}.json"
    model_file = Path(MODEL_CACHE_DIR) / f"model_{corpus_type}_ctx{context_size}.pt"
    
    return vocab_file, model_file


def main():
    """Entry point when script is run directly."""
    parser = argparse.ArgumentParser(description="Transformer Language Model")
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--nursery", action="store_true", help="Use nursery rhyme corpus")
    parser.add_argument("--context", type=int, default=8, help="Context window size")
    parser.add_argument("--topk", type=int, default=5, help="Show top-k predictions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--maxwords", type=int, default=5, help="Max number of predicted words")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--beam", type=int, default=0, help="Beam search width (0 disables beam search)")
    parser.add_argument("--force_train", action="store_true", help="Force retraining even if model exists")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    args = parser.parse_args()

    # Process arguments
    corpus_type = "nursery" if args.nursery else "shakespeare"
    
    # Load corpus and tokenize
    corpus_text = utils.load_corpus(corpus_type)
    tokens = utils.basic_tokenize(corpus_text)
    print(f"[INFO] Loaded corpus with {len(tokens)} tokens")
    
    # Get cache paths for model and vocabulary
    vocab_file, model_file = get_cache_paths(corpus_type, args.context)
    
    # Build or load vocabulary
    if vocab_file.exists() and not args.force_train:
        vocab = utils.load_vocab(vocab_file)
    else:
        special_tokens = ['<pad>', '<unk>', '<cls>', '<sep>']
        vocab = utils.build_vocab(tokens, min_freq=2, special_tokens=special_tokens)
        utils.save_vocab(vocab, vocab_file)

    # Create mapping from indices to words
    idx_to_word = {i: w for w, i in vocab.items()}
    
    # Encode the entire corpus
    encoded = utils.encode(tokens, vocab)
    
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
    if args.beam > 0:
        prediction, all_predictions, beams = predict_next(
            model, vocab, idx_to_word,
            args.prompt,
            context=args.context,
            top_k=args.topk,
            temperature=args.temperature,
            max_words=args.maxwords,
            beam_width=args.beam
        )
        
        # Print context info
        print(f"\nUsing Transformer model with context window size: {args.context} tokens")
        print(f"Beam search width: {args.beam}")
        
        # Print the complete prediction first
        print(f"\nTransformer Complete Prediction for '{args.prompt}': {prediction}")
        
        # Print predictions for each step
        for i, predictions in enumerate(all_predictions):
            step_prompt = args.prompt
            
            print(f"\nStep {i+1}: Top {len(predictions)} predictions:")
            for j, (word, prob) in enumerate(predictions, 1):
                print(f"  {j}. {word} (probability: {prob:.4f})")
        
        # Print final beam results
        print("\nFinal beam search results:")
        for i, (score, _, generated) in enumerate(sorted(beams, key=lambda x: x[0], reverse=True)[:args.topk]):
            print(f"  {i+1}. '{' '.join(generated)}' (score: {score:.4f})")
            
    else:
        prediction, all_predictions = predict_next(
            model, vocab, idx_to_word,
            args.prompt,
            context=args.context,
            top_k=args.topk,
            temperature=args.temperature,
            max_words=args.maxwords,
            beam_width=args.beam
        )
        
        # Print context info
        print(f"\nUsing Transformer model with context window size: {args.context} tokens")
        
        # Print the complete prediction first
        generated_words = prediction.split()
        print(f"\nTransformer Complete Prediction for '{args.prompt}': {prediction}")
        
        # Print predictions for each word
        for i, predictions in enumerate(all_predictions):
            current_word = generated_words[i] if i < len(generated_words) else "?"
            
            # Get the updated prompt for this step
            if i == 0:
                step_prompt = args.prompt
            else:
                step_prompt = f"{args.prompt} {' '.join(generated_words[:i])}"
                
            print(f"\nStep {i+1}: Top {len(predictions)} predictions for '{step_prompt}':")
            for j, (word, prob) in enumerate(predictions, 1):
                marker = "→" if word == current_word else " "
                print(f"  {marker} {j}. {word} (probability: {prob:.4f})")


if __name__ == "__main__":
    main()
