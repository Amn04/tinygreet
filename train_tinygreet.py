"""
Complete Training Script for TinyGreet

This script: 
1. Loads the tokenizer
2. Loads the dataset
3. Creates the model
4. Trains the model
5. Generates sample outputs
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
    
    # Model hyperparameters
    VOCAB_SIZE = 500
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    FF_HIDDEN_DIM = 256
    MAX_SEQ_LEN = 128
    DROPOUT_RATE = 0.1
    
    # Training hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    WARMUP_STEPS = 50
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
    
    generator = TextGenerator(model, tokenizer)
    
    test_prompts = [
        "Hello! ",
        "Good morning!",
        "How are you? ",
        "Goodbye!",
        "Thank you!",
    ]
    
    print("\n--- Greedy Decoding ---")
    for prompt in test_prompts:
        response = generator.generate_greedy(prompt)
        print(f"User: {prompt}")
        print(f"Bot:   {response}\n")
    
    print("\n--- With Sampling (temp=0.7, top_p=0.9) ---")
    for prompt in test_prompts[: 3]:
        responses = generator.generate(
            prompt,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=3
        )
        print(f"User:  {prompt}")
        for i, response in enumerate(responses):
            print(f"  Response {i+1}:  {response}")
        print()
    
    # ==================== INTERACTIVE CHAT ====================
    
    print("\n" + "=" * 70)
    print("Would you like to chat with TinyGreet?")
    print("=" * 70)
    
    try:
        answer = input("\nStart interactive chat? (y/n): ").strip().lower()
        if answer == 'y':
            interactive_chat(generator)
    except (EOFError, KeyboardInterrupt):
        pass
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {CHECKPOINT_DIR}/final_model/")
    print("\nTo use the trained model later:")
    print("  from model.tinygreet_model import TinyGreetModel")
    print("  model = TinyGreetModel.load('checkpoints/final_model')")


if __name__ == "__main__": 
    main()