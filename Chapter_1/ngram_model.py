import html
import re
import argparse
import requests
from collections import Counter
import nltk
from nltk import word_tokenize

# Explicitly download the required resources upfront
def ensure_nltk_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('gutenberg')

def load_corpus(use_nursery):
    if use_nursery:
        url = "https://www.gutenberg.org/files/38562/38562-h/38562-h.htm"
        html_content = requests.get(url).text
        text = re.sub(r"<.*?>", "", re.findall(r"<body.*?>(.*?)</body>", html_content, re.DOTALL)[0])
        text = html.unescape(text)
    else:
        from nltk.corpus import gutenberg
        text = gutenberg.raw('shakespeare-hamlet.txt')
    
    return word_tokenize(text.lower())

def build_ngram_model(tokens, n):
    model = {}
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model.setdefault(context, Counter())[next_word] += 1
    return model

def predict_next(model, prompt, n):
    tokens = word_tokenize(prompt.lower())
    print(f"[DEBUG] Prompt tokens: {tokens}")
    
    if len(tokens) < n - 1:
        return '<too-short>'
        
    context = tuple(tokens[-(n-1):])
    print(f"[DEBUG] Context used for prediction: {context}")
    
    candidates = model.get(context, {})
    if not candidates:
        return '<unk>'
        
    print(f"[DEBUG] Candidates: {dict(candidates)}")
    return candidates.most_common(1)[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Input prompt to complete")
    parser.add_argument("--nursery", action="store_true", help="Use nursery rhyme corpus")
    parser.add_argument("--ngram", type=int, default=4, help="Size of n-gram (default=4)")
    args = parser.parse_args()

    # Download required NLTK resources first
    ensure_nltk_resources()
    
    tokens = load_corpus(args.nursery)
    model = build_ngram_model(tokens, args.ngram)
    prediction = predict_next(model, args.prompt, args.ngram)
    print(f"N-gram Prediction for '{args.prompt}': {prediction}")

if __name__ == "__main__":
    main()
