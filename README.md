# TinyGreet 🤖

A tiny GPT-style language model built from scratch using only NumPy. No PyTorch, no TensorFlow - just pure Python and NumPy!

TinyGreet is designed for learning and experimenting with transformer architectures. It's a greeting/conversation chatbot that demonstrates all the core concepts of modern LLMs.

## Features

- **Transformer Architecture**: Multi-head attention, feed-forward networks, layer normalization
- **BPE Tokenizer**: Byte-Pair Encoding tokenizer trained from scratch
- **KV-Cache**: Efficient autoregressive generation
- **Beam Search**: Higher quality text generation
- **Repetition Penalty**: Reduces repetitive outputs
- **Conversation Context**: Multi-turn conversation support
- **Gradient Checkpointing**: Memory-efficient training for larger models

## Project Structure

```
tinygreet/
├── data/
│   ├── data_generator.py    # Generate training data
│   ├── build_dataset.py     # Build train/val/test splits
│   ├── final/               # Processed datasets
│   └── processed/           # Intermediate data
├── tokenizer/
│   ├── bpe.py               # BPE tokenizer implementation
│   ├── train_tokenizer.py   # Tokenizer training script
│   └── vocab/               # Saved tokenizer files
├── model/
│   ├── tensor.py            # Custom autograd Tensor class
│   ├── attention.py         # Attention mechanisms + KV-cache
│   ├── embeddings.py        # Token + positional embeddings
│   ├── feedforward.py       # Feed-forward network
│   ├── transformer_block.py # Transformer block + stack
│   ├── tinygreet_model.py   # Main model class
│   ├── trainer.py           # Training loop
│   ├── generate.py          # Text generation utilities
│   └── loss.py              # Cross-entropy loss
├── checkpoints/             # Saved model checkpoints
├── train_tinygreet.py       # Main training script
└── README.md
```

## Quick Start

### Prerequisites

```bash
pip install numpy tqdm
```

### Step 1: Generate Training Data

```bash
cd data
python3 data_generator.py
```

This creates `processed/generated_data.json` with greeting/response pairs.

### Step 2: Build Dataset

```bash
python3 build_dataset.py
```

This creates train/val/test splits in `final/` directory and a corpus for tokenizer training.

### Step 3: Train Tokenizer

```bash
cd ../tokenizer
python3 train_tokenizer.py
```

This trains a BPE tokenizer and saves it to `vocab/`.

### Step 4: Train the Model

```bash
cd ..
python3 train_tinygreet.py
```

Training will:
- Load the tokenizer and data
- Create the model (default: ~1.25M parameters)
- Train for configured epochs
- Save checkpoints to `checkpoints/`
- Test generation after training

### Step 5: Chat with the Model

After training, the script enters interactive chat mode. You can also load a trained model:

```python
import sys
sys.path.insert(0, 'model')
sys.path.insert(0, 'tokenizer')

from bpe import BPETokenizer
from tinygreet_model import TinyGreetModel
from generate import TextGenerator

# Load tokenizer and model
tokenizer = BPETokenizer.load('tokenizer/vocab/')
model = TinyGreetModel.load('checkpoints/best_model/')

# Create generator
generator = TextGenerator(model, tokenizer, use_kv_cache=True)

# Generate responses
response = generator.generate_greedy("Hello!", repetition_penalty=1.2)
print(response)

# Or use beam search for better quality
responses = generator.generate_beam_search("How are you?", beam_width=3)
print(responses[0])
```

## Adding More Training Data

### Option 1: Edit the Data Generator

Edit `data/data_generator.py` to add more patterns:

```python
def generate_your_category(self):
    """Add your own greeting category."""
    patterns = [
        # (input, output, category, subcategory, formality, time_of_day, mood)
        ("Your input", "Your response", "greeting", "custom", "casual", "any", "neutral"),
        # Add more patterns...
    ]
    
    for pattern in patterns:
        self.add_sample(*pattern)
```

Then call your function in `generate_all()`:

```python
def generate_all(self):
    # ... existing generators ...
    self.generate_your_category()  # Add this line
```

### Option 2: Add Data Directly to JSON

Add entries to `data/processed/generated_data.json`:

```json
{
  "input": "Your greeting",
  "output": "Your response",
  "metadata": {
    "category": "greeting",
    "subcategory": "custom",
    "formality": "casual",
    "time_of_day": "any",
    "mood": "neutral"
  }
}
```

### Option 3: Create a Custom Data File

Create your own JSON file and merge it:

```python
import json

# Load existing data
with open('data/processed/generated_data.json', 'r') as f:
    data = json.load(f)

# Add your data
custom_data = [
    {"input": "Howdy!", "output": "Howdy partner!", "metadata": {...}},
    # ... more entries
]
data.extend(custom_data)

# Save back
with open('data/processed/generated_data.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### After Adding Data

Rebuild the dataset and retrain:

```bash
cd data
python3 build_dataset.py
cd ../tokenizer
python3 train_tokenizer.py  # Optional: only if vocab changed significantly
cd ..
python3 train_tinygreet.py
```

## Configuration

Edit `train_tinygreet.py` to adjust hyperparameters:

```python
# Model size
EMBED_DIM = 128          # Embedding dimension
NUM_HEADS = 8            # Attention heads
NUM_LAYERS = 6           # Transformer layers
FF_HIDDEN_DIM = 512      # Feed-forward hidden size
MAX_SEQ_LEN = 256        # Maximum sequence length

# Training
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
NUM_EPOCHS = 30
WARMUP_STEPS = 100
```

### Model Size Presets

| Size | EMBED_DIM | HEADS | LAYERS | FF_DIM | Params |
|------|-----------|-------|--------|--------|--------|
| Tiny | 64 | 4 | 2 | 128 | ~150K |
| Small | 128 | 8 | 6 | 512 | ~1.25M |
| Medium | 256 | 8 | 8 | 1024 | ~10M |

## Generation Options

```python
# Greedy decoding (fast, deterministic)
response = generator.generate_greedy(prompt, repetition_penalty=1.2)

# Sampling (more creative)
responses = generator.generate(
    prompt,
    temperature=0.8,      # Higher = more random
    top_k=50,             # Only sample from top-k tokens
    top_p=0.9,            # Nucleus sampling
    repetition_penalty=1.2
)

# Beam search (highest quality)
responses = generator.generate_beam_search(
    prompt,
    beam_width=5,
    length_penalty=0.6,
    num_return_sequences=3
)
```

## Troubleshooting

### "Training data not found"
Run the data pipeline first:
```bash
cd data && python3 data_generator.py && python3 build_dataset.py
```

### "Tokenizer not found"
Train the tokenizer:
```bash
cd tokenizer && python3 train_tokenizer.py
```

### Out of Memory
Reduce model size or batch size in `train_tinygreet.py`:
```python
BATCH_SIZE = 4  # Reduce from 8
NUM_LAYERS = 4  # Reduce from 6
```

### Model generates repetitive text
Increase repetition penalty:
```python
generator.generate_greedy(prompt, repetition_penalty=1.5)
```

## License

MIT License - feel free to use, modify, and learn from this code!

## Acknowledgments

Built for learning purposes. Inspired by:
- Andrej Karpathy's nanoGPT
- The original Transformer paper ("Attention Is All You Need")
- Hugging Face Transformers
