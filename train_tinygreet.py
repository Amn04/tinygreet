"""
Complete Training Script for TinyGreet

This script: 
1. Loads the tokenizer
2. Loads the dataset
3. Creates the model
4. Trains the model
5. Generates sample outputs

Enhanced with:
- Scaled up model architecture
- More training epochs
- Beam search and repetition penalty in generation
- Conversation context support
- KV-Cache for faster generation
"""

import os
import sys
import numpy as np
import argparse

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from model.tensor import Tensor
from model.tinygreet_model import TinyGreetModel, TinyGreetConfig
from model.trainer import DataLoader, Trainer
from model.generate import TextGenerator, interactive_chat


def load_tokenizer(tokenizer_path: str):
    """Load the BPE tokenizer."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tokenizer'))
    from bpe import BPETokenizer
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    return tokenizer


def main():
    print("=" * 70)
    print("TINYGREET - Training from Scratch")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer', 'vocab')
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'final')
    TRAIN_DATA = os.path.join(DATA_DIR, 'train.json')
    VAL_DATA = os.path.join(DATA_DIR, 'val.json')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    
    # Model hyperparameters - SCALED UP for better capacity
    VOCAB_SIZE = 500
    EMBED_DIM = 128          # Increased from 64 - More expressive embeddings
    NUM_HEADS = 8            # Increased from 4 - More attention patterns
    NUM_LAYERS = 6           # Increased from 2 - Deeper model
    FF_HIDDEN_DIM = 512      # Increased from 256 - Larger feed-forward
    MAX_SEQ_LEN = 256        # Increased from 128 - Longer contexts
    DROPOUT_RATE = 0.1
    USE_GRADIENT_CHECKPOINTING = False  # Set to True for very large models
    
    # Training hyperparameters - ADJUSTED for more training
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4     # Slightly reduced for larger model
    NUM_EPOCHS = 30          # Reduced for faster testing
    WARMUP_STEPS = 100       # Increased warmup for larger model
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # ==================== LOAD TOKENIZER ====================
    
    print("\n📂 Loading tokenizer...")
    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        VOCAB_SIZE = tokenizer.get_vocab_size()
        print(f"   Vocabulary size: {VOCAB_SIZE}")
    except FileNotFoundError: 
        print("❌ Tokenizer not found!  Please train the tokenizer first:")
        print("   cd tokenizer && python train_tokenizer.py")
        return
    
    # ==================== LOAD DATA ====================
    
    print("\n📂 Loading training data...")
    try:
        train_loader = DataLoader(
            data_path=TRAIN_DATA,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            shuffle=True
        )
    except FileNotFoundError:
        print("❌ Training data not found! Please run the data pipeline first:")
        print("   cd data && python build_dataset.py")
        return
    
    print("\n📂 Loading validation data...")
    try: 
        val_loader = DataLoader(
            data_path=VAL_DATA,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            shuffle=False
        )
    except FileNotFoundError: 
        print("⚠️  Validation data not found, training without validation.")
        val_loader = None
    
    # ==================== CREATE MODEL ====================
    
    print("\n🔧 Creating model...")
    
    config = TinyGreetConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout_rate=DROPOUT_RATE,
        pad_token_id=tokenizer.vocab.get('<PAD>', 0),
        bos_token_id=tokenizer.vocab.get('<BOS>', 2),
        eos_token_id=tokenizer.vocab.get('<EOS>', 3),
        sep_token_id=tokenizer.vocab.get('<SEP>', 4),
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )
    
    model = TinyGreetModel(config)
    model.summary()
    
    # ==================== TRAIN ====================
    
    print("\n🚀 Starting training...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    trainer.train(
        num_epochs=NUM_EPOCHS,
        eval_every=1,
        save_every=2
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # ==================== TEST GENERATION ====================
    
    print("\n" + "=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)
    
    generator = TextGenerator(model, tokenizer, use_kv_cache=True)
    
    test_prompts = [
        "Hello! ",
        "Good morning!",
        "How are you? ",
        "Goodbye!",
        "Thank you!",
    ]
    
    print("\n--- Greedy Decoding (with repetition penalty) ---")
    for prompt in test_prompts:
        response = generator.generate_greedy(prompt, repetition_penalty=1.2)
        print(f"User: {prompt}")
        print(f"Bot:   {response}\n")
    
    print("\n--- With Sampling (temp=0.7, top_p=0.9, repetition_penalty=1.2) ---")
    for prompt in test_prompts[: 3]:
        responses = generator.generate(
            prompt,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=3,
            repetition_penalty=1.2
        )
        print(f"User:  {prompt}")
        for i, response in enumerate(responses):
            print(f"  Response {i+1}:  {response}")
        print()
    
    print("\n--- Beam Search (beam_width=4) ---")
    for prompt in test_prompts[:3]:
        responses = generator.generate_beam_search(
            prompt,
            beam_width=4,
            num_return_sequences=2,
            repetition_penalty=1.2
        )
        print(f"User: {prompt}")
        for i, response in enumerate(responses):
            print(f"  Beam {i+1}: {response}")
        print()
    
    # ==================== INTERACTIVE CHAT ====================
    
    print("\n" + "=" * 70)
    print("Would you like to chat with TinyGreet?")
    print("(Now with conversation context - the bot remembers previous turns!)")
    print("=" * 70)
    
    try:
        answer = input("\nStart interactive chat? (y/n): ").strip().lower()
        if answer == 'y':
            print("\nTip: The bot now remembers your conversation!")
            print("Type 'clear' to reset conversation history.")
            print("Type 'beam' to use beam search for higher quality responses.")
            print("-" * 60)
            interactive_chat_enhanced(generator)
    except (EOFError, KeyboardInterrupt):
        pass
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {CHECKPOINT_DIR}/final_model/")
    print("\nEnhancements in this version:")
    print("  - Scaled up model (128 embed_dim, 8 heads, 6 layers)")
    print("  - KV-Cache for faster generation")
    print("  - Beam search for better quality outputs")
    print("  - Repetition penalty to reduce repetitive text")
    print("  - Conversation context for multi-turn dialogue")
    print("  - Gradient checkpointing support for large models")
    print("\nTo use the trained model later:")
    print("  from model.tinygreet_model import TinyGreetModel")
    print("  model = TinyGreetModel.load('checkpoints/final_model')")


def interactive_chat_enhanced(generator):
    """Enhanced interactive chat with conversation context."""
    print("\n" + "=" * 60)
    print("TINYGREET INTERACTIVE CHAT (Enhanced)")
    print("=" * 60)
    print("\nType your greeting and I'll respond!")
    print("Commands:")
    print("  'quit' - Exit chat")
    print("  'clear' - Clear conversation history")
    print("  'beam' - Toggle beam search mode")
    print("  'settings' - Change generation settings")
    print("-" * 60)
    
    temperature = 0.7
    top_p = 0.9
    use_beam = False
    use_context = True
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'clear':
            generator.clear_conversation_history()
            print("Conversation history cleared!")
            continue
        
        if user_input.lower() == 'beam':
            use_beam = not use_beam
            print(f"Beam search: {'ON' if use_beam else 'OFF'}")
            continue
        
        if user_input.lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  Temperature: {temperature}")
            print(f"  Top-p: {top_p}")
            print(f"  Beam search: {'ON' if use_beam else 'OFF'}")
            print(f"  Use context: {'ON' if use_context else 'OFF'}")
            try:
                temp_input = input("New temperature (0.1-2.0, or Enter to keep): ").strip()
                if temp_input:
                    temperature = float(temp_input)
                
                top_p_input = input("New top_p (0.1-1.0, or Enter to keep): ").strip()
                if top_p_input:
                    top_p = float(top_p_input)
                
                context_input = input("Use context? (y/n, or Enter to keep): ").strip().lower()
                if context_input == 'y':
                    use_context = True
                elif context_input == 'n':
                    use_context = False
                
                print(f"Settings updated!")
            except ValueError:
                print("Invalid input, keeping current settings.")
            continue
        
        # Generate response
        if use_beam:
            responses = generator.generate_beam_search(
                prompt=user_input,
                beam_width=4,
                num_return_sequences=1,
                repetition_penalty=1.2,
                use_context=use_context
            )
            response = responses[0] if responses else "..."
        else:
            response = generator.chat(
                prompt=user_input,
                temperature=temperature,
                top_p=top_p,
                use_context=use_context
            )
        
        print(f"Bot: {response}")


if __name__ == "__main__": 
    main()