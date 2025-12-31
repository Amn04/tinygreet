"""
Byte-Pair Encoding (BPE) Implementation from Scratch

This implements the same algorithm used in: 
- GPT-2, GPT-3, GPT-4 (OpenAI)
- RoBERTa (Facebook)
- Many other modern LLMs

We build everything from first principles using only Python builtins and basic collections.
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import time


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer built from scratch. 
    
    The BPE algorithm: 
    1. Start with a character-level vocabulary
    2. Count frequency of adjacent token pairs
    3. Merge the most frequent pair into a new token
    4. Repeat until desired vocabulary size is reached
    
    This creates a vocabulary of subwords that balances: 
    - Vocabulary size (not too large)
    - Sequence length (not too long)
    - Ability to represent any text (no OOV)
    """
    
    # Special tokens - these are reserved and won't be merged
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # Beginning of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence
    SEP_TOKEN = "<SEP>"  # Separator (between input and output)
    
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
    
    def __init__(self, vocab_size:  int = 500):
        """
        Initialize the BPE tokenizer. 
        
        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        
        # Core data structures
        self.vocab: Dict[str, int] = {}          # token -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token
        self.merges: List[Tuple[str, str]] = []  # ordered list of merge rules
        self.merge_ranks: Dict[Tuple[str, str], int] = {}  # merge -> priority
        
        # Training statistics
        self.token_frequencies: Counter = Counter()
        self.trained = False
        
        # Pre-tokenization pattern (similar to GPT-2)
        # This splits text into chunks before BPE
        # Handles:  words, numbers, punctuation, whitespace
        self.pre_tokenize_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )
    
    def _initialize_vocab_with_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        self.vocab = {}
        self.inverse_vocab = {}
        
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def _get_base_vocab_from_corpus(self, texts: List[str]) -> Set[str]: 
        """
        Extract all unique characters from the corpus.
        These form the base vocabulary before any merges.
        """
        chars = set()
        for text in texts: 
            chars.update(set(text))
        return chars
    
    def _pre_tokenize(self, text: str) -> List[str]: 
        """
        Pre-tokenize text into chunks before applying BPE. 
        
        This is important because: 
        1. We don't want to merge across word boundaries initially
        2. It speeds up training
        3. It gives more meaningful subwords
        
        Example:
            "Hello, world!" -> ["Hello", ",", " world", "!"]
        """
        # Find all matches
        tokens = self.pre_tokenize_pattern. findall(text)
        return tokens
    
    def _tokenize_word_to_chars(self, word:  str) -> List[str]:
        """
        Convert a word into a list of characters. 
        This is the starting point before merges.
        
        Example:
            "hello" -> ['h', 'e', 'l', 'l', 'o']
        """
        return list(word)
    
    def _get_pair_frequencies(
        self, 
        tokenized_corpus: List[List[str]]
    ) -> Counter: 
        """
        Count the frequency of adjacent token pairs in the corpus.
        
        Args:
            tokenized_corpus: List of tokenized words (each word is a list of tokens)
        
        Returns: 
            Counter of (token1, token2) -> frequency
        
        Example:
            [['h','e','l','l','o'], ['h','e','l','l','o']]
            -> {('h','e'): 2, ('e','l'): 2, ('l','l'): 2, ('l','o'): 2}
        """
        pairs = Counter()
        
        for word_tokens in tokenized_corpus:
            # Count pairs in this word
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pairs[pair] += 1
        
        return pairs
    
    def _merge_pair_in_tokens(
        self,
        tokens: List[str],
        pair: Tuple[str, str],
        merged:  str
    ) -> List[str]: 
        """
        Merge all occurrences of a pair in a token list.
        
        Args:
            tokens: List of tokens ['h', 'e', 'l', 'l', 'o']
            pair:  Pair to merge ('l', 'l')
            merged:  Merged result 'll'
        
        Returns:
            New token list ['h', 'e', 'll', 'o']
        """
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            # Check if current position starts the pair
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2  # Skip both tokens
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    def _merge_pair_in_corpus(
        self,
        tokenized_corpus: List[List[str]],
        pair:  Tuple[str, str],
        merged: str
    ) -> List[List[str]]:
        """
        Apply a merge to the entire corpus.
        """
        return [
            self._merge_pair_in_tokens(word_tokens, pair, merged)
            for word_tokens in tokenized_corpus
        ]
    
    def train(
        self,
        texts: List[str],
        verbose: bool = True,
        min_frequency: int = 2
    ):
        """
        Train the BPE tokenizer on a corpus.
        
        This is the main training loop: 
        1. Pre-tokenize all text
        2. Convert to characters
        3. Iteratively merge most frequent pairs
        
        Args:
            texts: List of training texts
            verbose:  Whether to print progress
            min_frequency:  Minimum frequency for a merge to be considered
        """
        start_time = time. time()
        
        if verbose:
            print("=" * 60)
            print("BPE TOKENIZER TRAINING")
            print("=" * 60)
            print(f"\nTarget vocabulary size: {self.vocab_size}")
            print(f"Training corpus size: {len(texts)} texts")
            print(f"Total characters: {sum(len(t) for t in texts):,}")
        
        # Step 1: Initialize with special tokens
        if verbose:
            print("\n📝 Step 1: Initializing special tokens...")
        self._initialize_vocab_with_special_tokens()
        
        # Step 2: Pre-tokenize corpus
        if verbose:
            print("📝 Step 2: Pre-tokenizing corpus...")
        
        pre_tokenized = []
        word_freq = Counter()  # Track word frequencies for efficiency
        
        for text in texts: 
            chunks = self._pre_tokenize(text)
            for chunk in chunks: 
                word_freq[chunk] += 1
        
        if verbose:
            print(f"   Unique words/chunks: {len(word_freq):,}")
        
        # Step 3: Convert to character-level tokens
        if verbose:
            print("📝 Step 3: Converting to character-level tokens...")
        
        # Build initial vocabulary from characters
        base_chars = set()
        tokenized_words = {}  # word -> list of tokens
        
        for word in word_freq:
            char_tokens = self._tokenize_word_to_chars(word)
            tokenized_words[word] = char_tokens
            base_chars.update(char_tokens)
        
        # Add base characters to vocabulary
        next_id = len(self.vocab)
        for char in sorted(base_chars):  # Sort for determinism
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.inverse_vocab[next_id] = char
                next_id += 1
        
        if verbose:
            print(f"   Base character vocabulary: {len(base_chars)} chars")
            print(f"   Current vocab size: {len(self.vocab)}")
        
        # Step 4: Iteratively merge pairs
        if verbose: 
            print("\n📝 Step 4: Learning BPE merges...")
            print("-" * 60)
        
        num_merges = self.vocab_size - len(self.vocab)
        self.merges = []
        
        for merge_idx in range(num_merges):
            # Count pair frequencies (weighted by word frequency)
            pair_freq = Counter()
            for word, freq in word_freq. items():
                word_tokens = tokenized_words[word]
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_freq[pair] += freq
            
            if not pair_freq: 
                if verbose:
                    print(f"\n⚠️  No more pairs to merge at step {merge_idx}")
                break
            
            # Find most frequent pair
            best_pair = pair_freq.most_common(1)[0]
            pair, freq = best_pair
            
            # Check minimum frequency
            if freq < min_frequency: 
                if verbose: 
                    print(f"\n⚠️  Most frequent pair has freq {freq} < {min_frequency}, stopping")
                break
            
            # Create merged token
            merged = pair[0] + pair[1]
            
            # Add to vocabulary
            self.vocab[merged] = next_id
            self.inverse_vocab[next_id] = merged
            next_id += 1
            
            # Record the merge
            self.merges.append(pair)
            self.merge_ranks[pair] = len(self.merges) - 1
            
            # Apply merge to all words
            for word in tokenized_words: 
                tokenized_words[word] = self._merge_pair_in_tokens(
                    tokenized_words[word], pair, merged
                )
            
            # Progress logging
            if verbose and (merge_idx + 1) % 50 == 0:
                print(f"   Merge {merge_idx + 1:4d}/{num_merges}:  "
                      f"'{pair[0]}' + '{pair[1]}' -> '{merged}' (freq: {freq: ,})")
        
        # Store final statistics
        self.trained = True
        elapsed = time.time() - start_time
        
        if verbose: 
            print("-" * 60)
            print(f"\n✅ Training complete!")
            print(f"   Final vocabulary size: {len(self.vocab)}")
            print(f"   Number of merges learned: {len(self.merges)}")
            print(f"   Training time: {elapsed:.2f} seconds")
            
            # Show some example merges
            print("\n📋 Sample merges learned:")
            for i, (p1, p2) in enumerate(self.merges[:10]):
                print(f"   {i+1}. '{p1}' + '{p2}' -> '{p1}{p2}'")
            if len(self.merges) > 10:
                print(f"   ... and {len(self.merges) - 10} more")
    
    def encode(self, text: str, add_special_tokens:  bool = True) -> List[int]:
        """
        Encode text into token IDs. 
        
        Args:
            text:  Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns: 
            List of token IDs
        """
        if not self.trained:
            raise RuntimeError("Tokenizer must be trained before encoding!")
        
        # Pre-tokenize
        chunks = self._pre_tokenize(text)
        
        all_tokens = []
        
        # Add BOS token
        if add_special_tokens:
            all_tokens.append(self.vocab[self.BOS_TOKEN])
        
        # Process each chunk
        for chunk in chunks: 
            # Start with characters
            tokens = self._tokenize_word_to_chars(chunk)
            
            # Apply merges in order of priority
            # We need to iteratively apply merges until no more can be applied
            changed = True
            while changed:
                changed = False
                i = 0
                new_tokens = []
                
                while i < len(tokens):
                    # Try to find a merge for current position
                    if i < len(tokens) - 1:
                        pair = (tokens[i], tokens[i + 1])
                        if pair in self.merge_ranks:
                            # Merge this pair
                            new_tokens.append(tokens[i] + tokens[i + 1])
                            i += 2
                            changed = True
                            continue
                    
                    new_tokens.append(tokens[i])
                    i += 1
                
                tokens = new_tokens
            
            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab:
                    all_tokens.append(self.vocab[token])
                else:
                    all_tokens.append(self.vocab[self.UNK_TOKEN])
        
        # Add EOS token
        if add_special_tokens:
            all_tokens.append(self.vocab[self.EOS_TOKEN])
        
        return all_tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens:  bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids: 
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.UNK_TOKEN)
        
        return "".join(tokens)
    
    def encode_pair(
        self, 
        input_text: str, 
        output_text: str,
        add_special_tokens: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Encode an input-output pair (for training).
        
        Format: <BOS> input <SEP> output <EOS>
        
        Returns:
            Tuple of (input_ids, full_sequence_ids)
        """
        # Encode input (without special tokens)
        input_ids = self.encode(input_text, add_special_tokens=False)
        
        # Encode output (without special tokens)
        output_ids = self.encode(output_text, add_special_tokens=False)
        
        if add_special_tokens:
            # Full sequence: <BOS> input <SEP> output <EOS>
            full_sequence = (
                [self.vocab[self.BOS_TOKEN]] +
                input_ids +
                [self.vocab[self.SEP_TOKEN]] +
                output_ids +
                [self.vocab[self.EOS_TOKEN]]
            )
        else: 
            full_sequence = input_ids + output_ids
        
        return input_ids, full_sequence
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size."""
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self.vocab.get(token, self.vocab[self.UNK_TOKEN])
    
    def id_to_token(self, token_id: int) -> str:
        """Convert a token ID to its string."""
        return self.inverse_vocab.get(token_id, self.UNK_TOKEN)
    
    def save(self, directory: str):
        """
        Save the tokenizer to a directory.
        
        Saves:
        - vocab.json: Token to ID mapping
        - merges.txt: Merge rules in order
        - config.json: Tokenizer configuration
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os. path.join(directory, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges_path = os.path.join(directory, "merges.txt")
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version:  1.0\n")
            for p1, p2 in self.merges:
                # Escape special characters for readability
                p1_escaped = p1.replace('\n', '\\n').replace('\t', '\\t').replace(' ', '␣')
                p2_escaped = p2.replace('\n', '\\n').replace('\t', '\\t').replace(' ', '␣')
                f.write(f"{p1_escaped} {p2_escaped}\n")
        
        # Save config
        config_path = os.path.join(directory, "config.json")
        config = {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "special_tokens": self.SPECIAL_TOKENS,
            "trained": self.trained
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Saved tokenizer to {directory}/")
        print(f"   - vocab.json ({len(self.vocab)} tokens)")
        print(f"   - merges.txt ({len(self.merges)} merges)")
        print(f"   - config.json")
    
    @classmethod
    def load(cls, directory: str) -> 'BPETokenizer':
        """
        Load a tokenizer from a directory. 
        """
        import os
        
        # Load config
        config_path = os.path. join(directory, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(vocab_size=config["vocab_size"])
        
        # Load vocabulary
        vocab_path = os.path. join(directory, "vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokenizer.vocab = json.load(f)
        
        # Build inverse vocabulary
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        # Load merges
        merges_path = os.path. join(directory, "merges.txt")
        tokenizer.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line. startswith('#') or not line: 
                    continue
                parts = line.split(' ')
                if len(parts) == 2:
                    p1 = parts[0]. replace('\\n', '\n').replace('\\t', '\t').replace('␣', ' ')
                    p2 = parts[1].replace('\\n', '\n').replace('\\t', '\t').replace('␣', ' ')
                    tokenizer. merges.append((p1, p2))
        
        # Build merge ranks
        tokenizer.merge_ranks = {
            pair: i for i, pair in enumerate(tokenizer.merges)
        }
        
        tokenizer.trained = True
        
        print(f"📂 Loaded tokenizer from {directory}/")
        print(f"   - Vocabulary size: {len(tokenizer.vocab)}")
        print(f"   - Number of merges:  {len(tokenizer.merges)}")
        
        return tokenizer
    
    def print_vocabulary(self, max_tokens: int = 50):
        """Print a sample of the vocabulary."""
        print("\n📖 Vocabulary Sample:")
        print("-" * 40)
        
        # Group by token type
        special = []
        single_char = []
        multi_char = []
        
        for token, idx in sorted(self.vocab. items(), key=lambda x:  x[1]):
            if token in self.SPECIAL_TOKENS: 
                special.append((token, idx))
            elif len(token) == 1:
                single_char.append((token, idx))
            else:
                multi_char.append((token, idx))
        
        print(f"\nSpecial tokens ({len(special)}):")
        for token, idx in special: 
            print(f"  {idx:4d}: {repr(token)}")
        
        print(f"\nSingle characters ({len(single_char)}):")
        display_chars = single_char[:20]
        char_strs = [f"{repr(t)}:{i}" for t, i in display_chars]
        print(f"  {', '.join(char_strs)}")
        if len(single_char) > 20:
            print(f"  ...  and {len(single_char) - 20} more")
        
        print(f"\nLearned subwords ({len(multi_char)}):")
        for token, idx in multi_char[: max_tokens - len(special) - min(20, len(single_char))]:
            display_token = repr(token) if token. isspace() or not token.isprintable() else token
            print(f"  {idx: 4d}: {display_token}")
        if len(multi_char) > max_tokens:
            print(f"  ... and {len(multi_char) - max_tokens} more")


# ============================================================
# DEMONSTRATION AND TESTING
# ============================================================

def demonstrate_bpe():
    """Demonstrate BPE training and usage step by step."""
    
    print("=" * 70)
    print("BPE TOKENIZER DEMONSTRATION")
    print("=" * 70)
    
    # Simple corpus for demonstration
    demo_corpus = [
        "hello",
        "hello hello",
        "hello world",
        "hello there",
        "world world",
        "goodbye",
        "goodbye world",
    ]
    
    print("\n📚 Demo corpus:")
    for text in demo_corpus: 
        print(f"   '{text}'")
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(demo_corpus, verbose=True, min_frequency=1)
    
    # Test encoding/decoding
    print("\n" + "=" * 70)
    print("ENCODING/DECODING TEST")
    print("=" * 70)
    
    test_texts = [
        "hello",
        "hello world",
        "goodbye world",
        "hello there friend",  # New combination
    ]
    
    for text in test_texts: 
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Show tokens
        tokens = [tokenizer.id_to_token(i) for i in encoded]
        
        print(f"\n   Input:   '{text}'")
        print(f"   Tokens:   {tokens}")
        print(f"   IDs:     {encoded}")
        print(f"   Decoded: '{decoded}'")
    
    return tokenizer


if __name__ == "__main__":
    demonstrate_bpe()