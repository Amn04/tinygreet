"""
Training Infrastructure for TinyGreet

This includes:
- Data loading and batching
- Training loop
- Evaluation
- Checkpointing
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
import json
import os
import time
from tqdm import tqdm

from tensor import Tensor
from tinygreet_model import TinyGreetModel, TinyGreetConfig
from loss import CrossEntropyLoss, compute_perplexity
from optimizer import AdamW, LearningRateScheduler, clip_grad_norm


class DataLoader:
    """
    Data loader for language model training.
    
    Handles: 
    - Loading JSON data
    - Tokenizing with our BPE tokenizer
    - Creating input/target pairs
    - Batching
    - Shuffling
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int = 8,
        max_seq_len:  int = 128,
        shuffle: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Our BPE tokenizer
            batch_size: Number of sequences per batch
            max_seq_len:  Maximum sequence length (truncate longer sequences)
            shuffle: Whether to shuffle data each epoch
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        
        # Load data
        self.samples = self._load_data(data_path)
        print(f"📂 Loaded {len(self.samples)} samples from {data_path}")
        
        # Tokenize all samples
        self.tokenized_samples = self._tokenize_samples()
        print(f"   Tokenized {len(self.tokenized_samples)} samples")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load samples from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _tokenize_samples(self) -> List[np.ndarray]: 
        """Tokenize all samples into input/output pairs."""
        tokenized = []
        
        for sample in self.samples:
            # Get input and output text
            input_text = sample['input']
            output_text = sample['output']
            
            # Tokenize using our pair encoding format
            # Format: <BOS> input <SEP> output <EOS>
            _, full_ids = self.tokenizer.encode_pair(input_text, output_text)
            
            # Convert to numpy array
            token_ids = np.array(full_ids, dtype=np.int32)
            
            # Truncate if too long
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            
            tokenized.append(token_ids)
        
        return tokenized
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.tokenized_samples) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches. 
        
        Yields:
            Tuple of (input_ids, target_ids)
            - input_ids: (batch_size, seq_len)
            - target_ids: (batch_size, seq_len)
        """
        # Shuffle samples
        indices = np.arange(len(self.tokenized_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for batch_start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]
            batch_samples = [self.tokenized_samples[i] for i in batch_indices]
            
            # Pad to same length
            max_len = max(len(s) for s in batch_samples)
            max_len = min(max_len, self.max_seq_len)
            
            pad_id = self.tokenizer.vocab.get('<PAD>', 0)
            
            batch_padded = np.full((self.batch_size, max_len), pad_id, dtype=np.int32)
            for i, sample in enumerate(batch_samples):
                length = min(len(sample), max_len)
                batch_padded[i, :length] = sample[:length]
            
            # Create input and target (target is shifted by 1)
            input_ids = batch_padded[:, :-1]
            target_ids = batch_padded[:, 1:]
            
            yield input_ids, target_ids
    
    def get_sample_text(self, idx: int) -> Tuple[str, str]:
        """Get original text for a sample."""
        sample = self.samples[idx]
        return sample['input'], sample['output']


class Trainer:
    """
    Trainer for TinyGreet model. 
    
    Handles the complete training process:
    - Training loop
    - Validation
    - Logging
    - Checkpointing
    """
    
    def __init__(
        self,
        model: TinyGreetModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr:  float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize trainer.
        
        Args:
            model: The TinyGreet model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        # Create optimizer
        self.optimizer = AdamW(
            parameters=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Store config for scheduler (will be initialized when training starts)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.scheduler = None
        
        # Loss function
        self.loss_fn = CrossEntropyLoss(
            ignore_index=model.config.pad_token_id
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]: 
        """
        Train for one epoch. 
        
        Args:
            epoch:  Current epoch number
        
        Returns: 
            Dictionary with training metrics
        """
        self.model.training = True
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=True
        )
        
        for input_ids, target_ids in progress_bar:
            # Forward pass
            logits = self.model(input_ids, training=True)
            
            # Compute loss
            loss = self.loss_fn(logits, target_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                current_lr = self.scheduler.step()
            else:
                current_lr = self.lr
            
            # Track metrics
            loss_value = float(loss.data)
            total_loss += loss_value
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            perplexity = compute_perplexity(loss_value)
            progress_bar.set_postfix({
                'loss':  f'{loss_value:.4f}',
                'ppl': f'{perplexity:.2f}',
                'lr': f'{current_lr:.2e}',
                'grad':  f'{grad_norm:.2f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_perplexity = compute_perplexity(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }
    
    @np.errstate(over='ignore')  # Ignore overflow in exp
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns: 
            Dictionary with evaluation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.training = False
        
        total_loss = 0.0
        num_batches = 0
        
        for input_ids, target_ids in self.val_loader: 
            # Forward pass (no gradients needed)
            logits = self.model(input_ids, training=False)
            
            # Compute loss
            loss = self.loss_fn(logits, target_ids)
            
            total_loss += float(loss.data)
            num_batches += 1
        
        if num_batches == 0:
            return {}
        
        avg_loss = total_loss / num_batches
        avg_perplexity = compute_perplexity(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }
    
    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 1,
        verbose: bool = True
    ):
        """
        Full training loop.
        
        Args: 
            num_epochs:  Number of epochs to train
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
            verbose: Whether to print detailed logs
        """
        print("\n" + "=" * 60)
        print("TRAINING TINYGREET MODEL")
        print("=" * 60)
        
        # Calculate total steps for scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * num_epochs
        
        print(f"\nTraining configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Warmup steps: {self.warmup_steps}")
        
        # Initialize scheduler
        self.scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr / 10,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps
        )
        
        print(f"\nStarting training...")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_perplexity'].append(train_metrics['perplexity'])
            self.history['learning_rate'].append(self.scheduler.get_lr())
            
            # Evaluate
            if epoch % eval_every == 0 and self.val_loader is not None:
                val_metrics = self.evaluate()
                self.history['val_loss'].append(val_metrics.get('loss', 0))
                self.history['val_perplexity'].append(val_metrics.get('perplexity', 0))
                
                if verbose:
                    print(f"\n  Validation - Loss: {val_metrics['loss']:.4f}, "
                          f"Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss: 
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model')
                    if verbose:
                        print(f"  📝 New best model saved!")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}')
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nTraining time:  {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final train perplexity:  {self.history['train_perplexity'][-1]:.2f}")
        
        if self.val_loader is not None and self.history['val_loss']: 
            print(f"Best val loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model')
        print(f"\n💾 Final model saved to {self.checkpoint_dir}/final_model/")
    
    def save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'optimizer_state': self.optimizer.state_dict()
        }
        
        with open(os.path.join(checkpoint_path, 'training_state.pkl'), 'wb') as f:
            import pickle
            pickle.dump(training_state, f)
    
    def load_checkpoint(self, name: str):
        """Load a training checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        
        # Load model
        self.model = TinyGreetModel.load(checkpoint_path)
        
        # Load training state
        with open(os.path.join(checkpoint_path, 'training_state.pkl'), 'rb') as f:
            import pickle
            training_state = pickle.load(f)
        
        self.global_step = training_state['global_step']
        self.best_val_loss = training_state['best_val_loss']
        self.history = training_state['history']
        self.optimizer.load_state_dict(training_state['optimizer_state'])
        
        # Recreate optimizer with loaded model parameters
        self.optimizer = AdamW(
            parameters=self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        self.optimizer.load_state_dict(training_state['optimizer_state'])
    
    def plot_training_history(self):
        """Print training history (text-based plot)."""
        print("\n" + "=" * 60)
        print("TRAINING HISTORY")
        print("=" * 60)
        
        if not self.history['train_loss']:
            print("No training history available.")
            return
        
        # Print loss curve (ASCII art)
        losses = self.history['train_loss']
        max_loss = max(losses)
        min_loss = min(losses)
        
        print(f"\nTraining Loss (max={max_loss:.4f}, min={min_loss:.4f}):")
        print("-" * 50)
        
        height = 10
        width = min(len(losses), 50)
        
        # Sample losses if too many
        if len(losses) > width:
            indices = np.linspace(0, len(losses)-1, width, dtype=int)
            sampled_losses = [losses[i] for i in indices]
        else:
            sampled_losses = losses
        
        for row in range(height, 0, -1):
            threshold = min_loss + (max_loss - min_loss) * row / height
            line = ""
            for loss in sampled_losses:
                if loss >= threshold:
                    line += "█"
                else: 
                    line += " "
            print(f"  {threshold: 6.3f} |{line}|")
        
        print(f"         +" + "-" * len(sampled_losses) + "+")
        print(f"          Epoch 1{' ' * (len(sampled_losses) - 8)}Epoch {len(losses)}")


# ==================== TEST ====================
def test_data_loader():
    """Test data loader."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADER")
    print("=" * 60)
    
    # Create a mock tokenizer for testing
    class MockTokenizer: 
        def __init__(self):
            self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, '<SEP>': 4}
            for i in range(5, 100):
                self.vocab[f'token_{i}'] = i
        
        def encode_pair(self, input_text, output_text):
            # Simple mock encoding
            input_len = min(len(input_text.split()), 10)
            output_len = min(len(output_text.split()), 10)
            
            input_ids = list(range(5, 5 + input_len))
            output_ids = list(range(20, 20 + output_len))
            
            full_ids = [2] + input_ids + [4] + output_ids + [3]
            return input_ids, full_ids
    
    # Create mock data file
    mock_data = [
        {"input": "Hello", "output": "Hi there how are you"},
        {"input": "Good morning", "output": "Good morning to you too"},
        {"input": "Bye", "output": "Goodbye take care"},
    ] * 10  # Repeat to have enough samples
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_data, f)
        temp_path = f.name
    
    try: 
        tokenizer = MockTokenizer()
        loader = DataLoader(
            data_path=temp_path,
            tokenizer=tokenizer,
            batch_size=4,
            max_seq_len=32
        )
        
        print(f"\nNumber of batches: {len(loader)}")
        
        # Get first batch
        for input_ids, target_ids in loader: 
            print(f"\nFirst batch:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Target shape:  {target_ids. shape}")
            print(f"  Input[0]: {input_ids[0]}")
            print(f"  Target[0]: {target_ids[0]}")
            break
        
        print("\n✅ Data loader tests passed!")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_data_loader()
