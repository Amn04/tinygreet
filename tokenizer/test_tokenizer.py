"""
Interactive Tokenizer Testing

Run this to interactively test the trained tokenizer.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os. path.abspath(__file__)))

from bpe import BPETokenizer


def interactive_mode(tokenizer:  BPETokenizer):
    """Interactive tokenizer testing."""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE TOKENIZER TEST")
    print("=" * 60)
    print("\nEnter text to tokenize (or 'quit' to exit)")
    print("Commands:")
    print("  'vocab'  - Show vocabulary sample")
    print("  'stats'  - Show tokenizer statistics")
    print("  'quit'   - Exit")
    print("-" * 60)
    
    while True:
        try:
            text = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not text:
            continue
        
        if text.lower() == 'quit': 
            break
        
        if text.lower() == 'vocab': 
            tokenizer.print_vocabulary(max_tokens=50)
            continue
        
        if text.lower() == 'stats': 
            print(f"\nVocabulary size: {tokenizer. get_vocab_size()}")
            print(f"Number of merges: {len(tokenizer.merges)}")
            print(f"Special tokens: {tokenizer. SPECIAL_TOKENS}")
            continue
        
        # Tokenize the input
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer. decode(encoded)
            tokens = [tokenizer. id_to_token(t) for t in encoded]
            
            print(f"\n   Input:    '{text}'")
            print(f"   Tokens:   {tokens}")
            print(f"   IDs:      {encoded}")
            print(f"   Decoded:  '{decoded}'")
            print(f"   Length:   {len(encoded)} tokens")
            
            # Show compression
            if len(encoded) > 0:
                ratio = len(text) / len(encoded)
                print(f"   Compression: {ratio:.2f} chars/token")
            
        except Exception as e:
            print(f"\n   ❌ Error:  {e}")
    
    print("\nGoodbye!")


def run_test_suite(tokenizer: BPETokenizer):
    """Run a comprehensive test suite."""
    
    print("\n" + "=" * 60)
    print("TOKENIZER TEST SUITE")
    print("=" * 60)
    
    test_cases = [
        # Basic greetings
        ("hello", "Basic greeting"),
        ("Hello!", "Greeting with punctuation"),
        ("HELLO", "Uppercase greeting"),
        
        # Time-based
        ("Good morning!", "Morning greeting"),
        ("Good afternoon", "Afternoon greeting"),
        ("Good evening!", "Evening greeting"),
        ("Good night", "Night greeting"),
        
        # Casual
        ("Hey!  What's up?", "Casual greeting"),
        ("Yo!  How's it going?", "Very casual"),
        ("Sup?", "Slang"),
        
        # Farewells
        ("Goodbye!", "Basic farewell"),
        ("See you later!", "Casual farewell"),
        ("Take care!", "Warm farewell"),
        
        # Longer texts
        ("Hello!  How are you doing today?", "Long greeting"),
        ("Good morning! I hope you have a wonderful day!", "Long morning greeting"),
        
        # Edge cases
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("!! !", "Punctuation only"),
        ("Hello... ??? ", "Mixed punctuation"),
        
        # Numbers and special chars
        ("Hi!  It's 9am.", "With numbers"),
        ("Hello @everyone!", "With special char"),
    ]
    
    passed = 0
    failed = 0
    
    for text, description in test_cases: 
        try: 
            if text. strip():  # Skip empty/whitespace for encoding test
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                
                # Basic validation
                assert len(encoded) > 0, "Encoding produced no tokens"
                assert all(isinstance(t, int) for t in encoded), "Not all tokens are ints"
                
                status = "✅"
                passed += 1
            else:
                encoded = tokenizer.encode(text) if text else []
                decoded = tokenizer. decode(encoded) if encoded else ""
                status = "✅"
                passed += 1
                
        except Exception as e: 
            status = f"❌ {e}"
            failed += 1
        
        tokens = [tokenizer. id_to_token(t) for t in encoded] if 'encoded' in dir() and encoded else []
        print(f"\n{status} {description}")
        print(f"   Input:  '{text}'")
        print(f"   Tokens: {tokens[: 10]}{'...' if len(tokens) > 10 else ''}")
    
    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


def main():
    """Main test runner."""
    
    # Try to load existing tokenizer
    vocab_dir = "vocab"
    
    if os.path.exists(os.path.join(vocab_dir, "config.json")):
        print("📂 Loading trained tokenizer...")
        tokenizer = BPETokenizer.load(vocab_dir)
    else:
        print("⚠️  No trained tokenizer found!")
        print("   Please run train_tokenizer.py first.")
        print("\n   Creating a demo tokenizer for testing...")
        
        # Create demo tokenizer
        demo_corpus = [
            "hello", "hello!", "Hello there! ",
            "good morning", "good afternoon", "good evening",
            "goodbye", "bye", "see you later",
            "how are you", "I'm fine thanks",
            "thank you", "you're welcome",
        ]
        
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(demo_corpus, verbose=True, min_frequency=1)
    
    # Run test suite
    print("\n" + "=" * 60)
    print("Running test suite...")
    all_passed = run_test_suite(tokenizer)
    
    # Interactive mode
    if all_passed:
        response = input("\n\nRun interactive mode? (y/n): ").strip().lower()
        if response == 'y': 
            interactive_mode(tokenizer)


if __name__ == "__main__": 
    main()