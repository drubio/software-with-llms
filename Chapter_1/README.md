# Language Model evolution

This chapter provides three different implementations of language models to compare their effectiveness on text prediction tasks:

- **N-gram Model** (`ngram_model.py`)
- **LSTM Model** (`lstm_model.py`)
- **Transformer Model** (`transformer_model.py`)

All models by default run on a Shakespeare corpus or can also use a nursery rhyme corpus (with the --nursery flag), to generate word predictions based on a given prompt.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

All scripts are command-line runnable and use default values for every parameter. Run the script with --help to see all the available parameters that can be overriden with flags

Below are example commands for each:

### N-gram

```bash
python ngram_model.py 'A rose by any'
python ngram_model.py "Twinkle, Twinkle, Little" --nursery --ngram 4 --topk 5
```

### LSTM

```bash
python lstm_model.py 'A rose by any' --context 15 --maxwords 5
python lstm_model.py "Twinkle, Twinkle, Little" --nursery --context 15 --epochs 10 --maxwords 5
```

### Transformer

```bash
python transformer_model.py 'A rose by any' --context 15 --maxwords 5
python transformer_model.py "Twinkle, Twinkle, Little" --nursery --context 15 --epochs 5 --maxwords 5
```

## Notes

- Models and vocabularies are cached to disk after training.
- The nursery corpus is downloaded from Project Gutenberg and Poetry Foundation if not cached.
- Transformer and LSTM models require training unless previously saved models are available.
- Use `--force_train` to retrain even if a cached model exists.

## Directory Structure

- `utils.py`: Shared functions for tokenization, vocab building, and corpus loading.
- `corpus_cache/`: Stores downloaded and cleaned corpus files.
- `vocab_cache/`: Caches vocabularies built from the corpus.
- `lstm_models/`, `transformer_models/`: Store trained model weights.
