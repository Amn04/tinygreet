"""
Train BPE Tokenizer on TinyGreet Dataset

This script: 
1. Loads our processed dataset
2. Trains a BPE tokenizer
3. Validates the tokenizer
4. Saves the trained tokenizer
"""

import os
import sys
import json
from typing import List

# Add parent directory to path for imports
sys.path. insert(0, os. path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpe import BPETokenizer


def load_corpus(data_dir: str = "../data/final") -> List[str]: 
    """Load training corpus from processed data."""
    
    corpus = []
    
    # Load from all_data.json
    all_data_path = os.path.join(data_dir, "all_data.json")
    
    if os.path.exists(all_data_path):
        print(f"📂 Loading data from {all_data_path}")
        with open(all_data_path, 'r', encoding='utf-8') as f:
            data = json. load(f)
        
        for sample in data:
            # Add both input and output to corpus
            corpus.append(sample["input"])
            corpus.append(sample["output"])
            # Also add the combined format (what the model will see)
            corpus.append(f"{sample['input']} ||| {sample['output']}")
    else:
        # Fallback to corpus. txt
        corpus_path = os. path.join(data_dir, "corpus.txt")
        if os.path. exists(corpus_path):
            print(f"📂 Loading data from {corpus_path}")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus = f.read().strip().split('\n')
        else: 
            raise FileNotFoundError(
                f"No data found!  Please run data pipeline first.\n"
                f"Expected:  {all_data_path} or {corpus_path}"
            )
    
    print(f"   Loaded {len(corpus)} text samples")
    return corpus


def analyze_corpus(corpus: List[str]):
    """Analyze corpus to determine optimal vocab size."""
    
    print("\n📊 Corpus Analysis:")
    print("-" * 50)
    
    # Character statistics
    all_text = " ".join(corpus)
    char_freq = {}
    for char in all_text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Word statistics
    words = all_text.lower().split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq. get(word, 0) + 1
    
    print(f"   Total characters: {len(all_text):,}")
    print(f"   Unique characters: {len(char_freq)}")
    print(f"   Total words: {len(words):,}")
    print(f"   Unique words: {len(word_freq)}")
    
    # Suggest vocab size
    # Rule of thumb: vocab_size ≈ 2 * sqrt(unique_words) for small corpora
    suggested_vocab = min(
        max(256, int(2 * (len(word_freq) ** 0.5))),  # At least 256
        1000  # Cap at 1000 for our small model
    )
    
    print(f"\n   Suggested vocab size: {suggested_vocab}")
    print(f"   (Based on corpus statistics)")
    
    # Show most common characters
    print("\n   Top 10 characters:")
    sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])[:10]
    for char, freq in sorted_chars:
        display = repr(char) if not char.isprintable() or char.isspace() else f"'{char}'"
        print(f"      {display}:  {freq: ,}")
    
    # Show most common words
    print("\n   Top 10 words:")
    sorted_words = sorted(word_freq.items(), key=lambda x:  -x[1])[:10]
    for word, freq in sorted_words: 
        print(f"      '{word}': {freq}")
    
    return suggested_vocab


def validate_tokenizer(tokenizer: BPETokenizer, test_samples: List[str]):
    """Validate the trained tokenizer on sample texts."""
    
    print("\n" + "=" * 60)
    print("TOKENIZER VALIDATION")
    print("=" * 60)
    
    total_tokens = 0
    total_chars = 0
    issues = []
    
    for i, text in enumerate(test_samples[: 20]):
        # Encode
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            
            total_tokens += len(encoded)
            total_chars += len(text)
            
            # Check for reconstruction issues
            # Note: We expect some differences due to special tokens
            decoded_clean = decoded. strip()
            text_clean = text. strip()
            
            if decoded_clean != text_clean: 
                issues.append({
                    "original": text,
                    "decoded": decoded_clean,
                    "encoded": encoded
                })
            
            # Show first few examples
            if i < 5:
                tokens = [tokenizer. id_to_token(t) for t in encoded]
                print(f"\n   Example {i+1}:")
                print(f"   Input:   '{text}'")
                print(f"   Tokens: {tokens}")
                print(f"   IDs:    {encoded}")
                print(f"   Output:  '{decoded}'")
                
        except Exception as e:
            issues.append({
                "original": text,
                "error": str(e)
            })
    
    # Summary statistics
    print("\n" + "-" * 60)
    print("VALIDATION SUMMARY")
    print("-" * 60)
    
    avg_tokens = total_tokens / len(test_samples[: 20])
    avg_chars = total_chars / len(test_samples[:20])
    compression = avg_chars / avg_tokens if avg_tokens > 0 else 0
    
    print(f"   Samples tested: {min(20, len(test_samples))}")
    print(f"   Average tokens per sample: {avg_tokens:.1f}")
    print(f"   Average chars per sample: {avg_chars:.1f}")
    print(f"   Compression ratio: {compression:.2f} chars/token")
    print(f"   Issues found: {len(issues)}")
    
    if issues: 
        print("\n   ⚠️  Samples with issues:")
        for issue in issues[: 3]:
            if "error" in issue: 
                print(f"      Error: {issue['error']}")
            else:
                print(f"      Original: '{issue['original'][: 50]}...'")
                print(f"      Decoded:   '{issue['decoded'][:50]}...'")
    
    return len(issues) == 0


def test_pair_encoding(tokenizer: BPETokenizer):
    """Test encoding of input-output pairs."""
    
    print("\n" + "=" * 60)
    print("PAIR ENCODING TEST")
    print("=" * 60)
    
    test_pairs = [
        ("Hello!", "Hi there!  How are you?"),
        ("Good morning", "Good morning! Have a great day! "),
        ("Bye!", "Goodbye! Take care!"),
    ]
    
    for input_text, output_text in test_pairs:
        input_ids, full_ids = tokenizer.encode_pair(input_text, output_text)
        
        full_tokens = [tokenizer. id_to_token(t) for t in full_ids]
        
        print(f"\n   Input:  '{input_text}'")
        print(f"   Output:  '{output_text}'")
        print(f"   Full tokens: {full_tokens}")
        print(f"   Full IDs:  {full_ids}")
        print(f"   Sequence length: {len(full_ids)}")


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("TinyGreet BPE Tokenizer Training")
    print("=" * 70)
    
    # Configuration
    VOCAB_SIZE = 500  # Start with 500, can adjust
    MIN_FREQUENCY = 2  # Minimum pair frequency to merge
    SAVE_DIR = "vocab"
    DATA_DIR = "../data/final"
    
    # Step 1: Load corpus
    print("\n📝 Step 1: Loading corpus...")
    try:
        corpus = load_corpus(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ Error:  {e}")
        print("\nPlease run the data pipeline first:")
        print("   cd data")
        print("   python build_dataset.py")
        return
    
    # Step 2: Analyze corpus
    print("\n📝 Step 2: Analyzing corpus...")
    suggested_vocab = analyze_corpus(corpus)
    
    # Use suggested or configured vocab size
    actual_vocab_size = max(VOCAB_SIZE, suggested_vocab)
    print(f"\n   Using vocab size: {actual_vocab_size}")
    
    # Step 3: Train tokenizer
    print("\n📝 Step 3: Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=actual_vocab_size)
    tokenizer.train(corpus, verbose=True, min_frequency=MIN_FREQUENCY)
    
    # Step 4: Validate
    print("\n📝 Step 4: Validating tokenizer...")
    validate_tokenizer(tokenizer, corpus[: 100])
    
    # Step 5: Test pair encoding
    test_pair_encoding(tokenizer)
    
    # Step 6: Print vocabulary sample
    tokenizer.print_vocabulary(max_tokens=30)
    
    # Step 7: Save tokenizer
    print("\n📝 Step 7: Saving tokenizer...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer. save(SAVE_DIR)
    
    # Step 8: Test loading
    print("\n📝 Step 8: Testing load/save cycle...")
    loaded_tokenizer = BPETokenizer. load(SAVE_DIR)
    
    # Verify loaded tokenizer works
    test_text = "Hello! How are you?"
    original_encoded = tokenizer.encode(test_text)
    loaded_encoded = loaded_tokenizer.encode(test_text)
    
    if original_encoded == loaded_encoded:
        print("   ✅ Load/save verification passed!")
    else: 
        print("   ❌ Load/save verification failed!")
        print(f"   Original: {original_encoded}")
        print(f"   Loaded:   {loaded_encoded}")
    
    print("\n" + "=" * 70)
    print("✅ TOKENIZER TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTokenizer saved to:  {os.path.abspath(SAVE_DIR)}/")
    print("\nNext steps:")
    print("   1. Review the vocabulary in vocab/vocab.json")
    print("   2. Review the merges in vocab/merges.txt")
    print("   3. Proceed to Phase 2: Building Embeddings!")


if __name__ == "__main__": 
    main()