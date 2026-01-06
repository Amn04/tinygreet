"""
Text Generation for TinyGreet

Implements various decoding strategies: 
1.  Greedy Decoding - Always pick the most likely token
2. Temperature Sampling - Add randomness based on temperature
3. Top-k Sampling - Sample from top k tokens
4. Top-p (Nucleus) Sampling - Sample from smallest set with cumulative prob >= p
5. Beam Search - Explore multiple hypotheses for better quality
6. Repetition Penalty - Reduce repetitive outputs
7. KV-Cache - Efficient autoregressive generation
8. Conversation Context - Multi-turn dialogue support
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

from tensor import Tensor
from tinygreet_model import TinyGreetModel
from attention import KVCache


class ConversationHistory:
    """
    Manages conversation history for multi-turn dialogue.
    Keeps track of previous turns to provide context.
    """
    
    def __init__(self, max_turns: int = 5, max_tokens: int = 512):
        """
        Initialize conversation history.
        
        Args:
            max_turns: Maximum number of turns to remember
            max_tokens: Maximum total tokens in history
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, user_input: str, bot_response: str):
        """Add a conversation turn."""
        self.history.append({
            'user': user_input,
            'bot': bot_response
        })
        # Keep only the last max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_context(self, tokenizer, max_tokens: int = None) -> str:
        """
        Get formatted conversation context.
        
        Returns context string that can be prepended to current input.
        """
        max_tokens = max_tokens or self.max_tokens
        
        context_parts = []
        for turn in self.history:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Bot: {turn['bot']}")
        
        context = " ".join(context_parts)
        
        # Truncate if too long (simple truncation from the start)
        # In production, you'd want smarter truncation
        tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > max_tokens:
            # Keep the most recent context
            tokens = tokens[-max_tokens:]
            context = tokenizer.decode(tokens)
        
        return context
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def __len__(self):
        return len(self.history)


class TextGenerator: 
    """
    Text generator for TinyGreet model. 
    
    Handles autoregressive generation with various sampling strategies.
    Now with KV-cache support, repetition penalty, beam search, and conversation context.
    """
    
    def __init__(
        self,
        model: TinyGreetModel,
        tokenizer,
        max_length: int = 50,
        use_kv_cache: bool = True
    ):
        """
        Initialize generator.
        
        Args:
            model:  Trained TinyGreet model
            tokenizer: BPE tokenizer
            max_length:  Maximum generation length
            use_kv_cache: Whether to use KV caching for faster generation
        """
        self.model = model
        self. tokenizer = tokenizer
        self.max_length = max_length
        self.use_kv_cache = use_kv_cache
        
        # Special token IDs
        self. bos_id = model.config.bos_token_id
        self.eos_id = model. config.eos_token_id
        self.sep_id = model. config.sep_token_id
        self.pad_id = model. config.pad_token_id
        
        # Conversation history for multi-turn support
        self.conversation_history = ConversationHistory()
    
    def _apply_repetition_penalty(
        self, 
        logits: np.ndarray, 
        token_ids: List[int],
        penalty: float = 1.2
    ) -> np.ndarray:
        """
        Apply repetition penalty to logits.
        
        Tokens that have already appeared get their logits divided by penalty.
        This discourages the model from repeating the same tokens.
        
        Args:
            logits: Raw logits from model, shape (vocab_size,)
            token_ids: List of previously generated token IDs
            penalty: Penalty factor (1.0 = no penalty, higher = more penalty)
        
        Returns:
            Modified logits
        """
        if penalty == 1.0:
            return logits
        
        # Get unique tokens that have appeared
        seen_tokens = set(token_ids)
        
        for token_id in seen_tokens:
            if token_id < len(logits):
                # Divide positive logits by penalty, multiply negative by penalty
                if logits[token_id] > 0:
                    logits[token_id] = logits[token_id] / penalty
                else:
                    logits[token_id] = logits[token_id] * penalty
        
        return logits
    
    def _get_logits(
        self, 
        token_ids: np.ndarray,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0
    ) -> np.ndarray:
        """
        Get logits for next token prediction.
        
        Args: 
            token_ids:  Current sequence of token IDs
            kv_cache: Optional KV cache for efficient generation
            position_offset: Position offset for cached generation
        
        Returns: 
            Logits for next token, shape (vocab_size,)
        """
        use_cache = kv_cache is not None and self.use_kv_cache
        
        # Forward pass through model
        logits = self.model(
            token_ids, 
            training=False,
            use_cache=use_cache,
            kv_cache=kv_cache,
            position_offset=position_offset
        )
        
        # Get logits for last position
        if logits.data.ndim == 2:
            return logits. data[-1]  # (vocab_size,)
        else:
            return logits.data[0, -1]  # (vocab_size,)
    
    def _sample_token(
        self,
        logits:  np.ndarray,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> int:
        """
        Sample a token from logits using specified strategy.
        
        Args:
            logits: Logits for all tokens, shape (vocab_size,)
            temperature: Temperature for sampling (lower = more deterministic)
            top_k: If > 0, only sample from top k tokens
            top_p: If < 1.0, use nucleus sampling
        
        Returns: 
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, len(probs))
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / np.sum(probs)
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[: :-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff
            cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_probs))
            
            # Zero out tokens beyond cutoff
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / np. sum(probs)
        
        # Sample
        token_id = np. random.choice(len(probs), p=probs)
        
        return int(token_id)
    
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int = None,
        repetition_penalty: float = 1.0,
        use_context: bool = False
    ) -> str:
        """
        Generate text using greedy decoding. 
        
        Always picks the most likely next token.
        Fast but can be repetitive.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            use_context: Whether to use conversation history
        
        Returns:
            Generated text (response only)
        """
        max_new_tokens = max_new_tokens or self.max_length
        
        # Prepare prompt with optional conversation context
        if use_context and len(self.conversation_history) > 0:
            context = self.conversation_history.get_context(self.tokenizer, max_tokens=100)
            full_prompt = f"{context} User: {prompt}"
        else:
            full_prompt = prompt
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Start with BOS + prompt + SEP
        token_ids = [self.bos_id] + prompt_ids + [self.sep_id]
        
        # Initialize KV cache if enabled
        kv_cache = self.model.create_kv_cache() if self.use_kv_cache else None
        
        # Process the prompt first to fill the cache and get initial logits
        logits = self._get_logits(np.array(token_ids), kv_cache=kv_cache)
        
        # Generate tokens
        for step in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, token_ids, repetition_penalty)
            
            # Pick highest probability token
            next_token = int(np.argmax(logits))
            
            # Stop if EOS
            if next_token == self.eos_id:
                break
            
            token_ids.append(next_token)
            
            # Get logits for next iteration
            if kv_cache is not None:
                # Only process the last token (use cached K,V)
                logits = self._get_logits(
                    np.array([next_token]), 
                    kv_cache=kv_cache,
                    position_offset=len(token_ids) - 1
                )
            else:
                # No cache - reprocess everything
                logits = self._get_logits(np.array(token_ids), kv_cache=None)
        
        # Decode response (everything after SEP)
        if self.sep_id in token_ids: 
            sep_pos = token_ids. index(self.sep_id)
            response_ids = token_ids[sep_pos + 1:]
        else:
            response_ids = token_ids
        
        # Filter out special tokens
        response_ids = [t for t in response_ids if t not in 
                       [self.bos_id, self.eos_id, self. sep_id, self.pad_id]]
        
        response = self.tokenizer.decode(response_ids)
        
        # Update conversation history if using context
        if use_context:
            self.conversation_history.add_turn(prompt, response)
        
        return response
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p:  float = 1.0,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2,
        use_context: bool = False
    ) -> List[str]:
        """
        Generate text with sampling.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature:  Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling parameter (0 = disabled)
            top_p:  Nucleus sampling parameter (1. 0 = disabled)
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            use_context: Whether to use conversation history
        
        Returns: 
            List of generated texts
        """
        max_new_tokens = max_new_tokens or self.max_length
        
        # Prepare prompt with optional conversation context
        if use_context and len(self.conversation_history) > 0:
            context = self.conversation_history.get_context(self.tokenizer, max_tokens=100)
            full_prompt = f"{context} User: {prompt}"
        else:
            full_prompt = prompt
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        
        results = []
        
        for _ in range(num_return_sequences):
            # Start with BOS + prompt + SEP
            token_ids = [self.bos_id] + prompt_ids + [self.sep_id]
            
            # Initialize KV cache for this sequence
            kv_cache = self.model.create_kv_cache() if self.use_kv_cache else None
            
            # Process the prompt first to fill the cache and get initial logits
            logits = self._get_logits(np.array(token_ids), kv_cache=kv_cache)
            
            # Generate tokens
            for step in range(max_new_tokens):
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits.copy(), token_ids, repetition_penalty)
                
                # Sample next token
                next_token = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Stop if EOS
                if next_token == self. eos_id: 
                    break
                
                token_ids.append(next_token)
                
                # Get logits for next iteration
                if kv_cache is not None:
                    # Only process the last token
                    logits = self._get_logits(
                        np.array([next_token]),
                        kv_cache=kv_cache,
                        position_offset=len(token_ids) - 1
                    )
                else:
                    logits = self._get_logits(np.array(token_ids), kv_cache=None)
            
            # Decode response
            if self. sep_id in token_ids:
                sep_pos = token_ids.index(self.sep_id)
                response_ids = token_ids[sep_pos + 1:]
            else:
                response_ids = token_ids
            
            # Filter out special tokens
            response_ids = [t for t in response_ids if t not in 
                           [self. bos_id, self.eos_id, self.sep_id, self. pad_id]]
            
            results. append(self.tokenizer.decode(response_ids))
        
        # Update conversation history if using context (only add first response)
        if use_context and results:
            self.conversation_history.add_turn(prompt, results[0])
        
        return results
    
    def chat(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_context: bool = True
    ) -> str:
        """
        Simple chat interface with conversation context.
        
        Args: 
            prompt: User input
            temperature:  Sampling temperature
            top_p:  Nucleus sampling parameter
            use_context: Whether to use conversation history
        
        Returns: 
            Model response
        """
        responses = self.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            repetition_penalty=1.2,
            use_context=use_context
        )
        return responses[0]
    
    def generate_beam_search(
        self,
        prompt: str,
        max_new_tokens: int = None,
        beam_width: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        use_context: bool = False
    ) -> List[str]:
        """
        Generate text using beam search for higher quality outputs.
        
        Beam search maintains multiple hypotheses (beams) and expands them
        in parallel, keeping the top-k best sequences at each step.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            beam_width: Number of beams to maintain
            length_penalty: Penalty for shorter sequences (< 1.0 favors shorter)
            repetition_penalty: Penalty for repeated tokens
            num_return_sequences: Number of top sequences to return
            use_context: Whether to use conversation history
        
        Returns:
            List of top generated sequences
        """
        max_new_tokens = max_new_tokens or self.max_length
        num_return_sequences = min(num_return_sequences, beam_width)
        
        # Prepare prompt with optional conversation context
        if use_context and len(self.conversation_history) > 0:
            context = self.conversation_history.get_context(self.tokenizer, max_tokens=100)
            full_prompt = f"{context} User: {prompt}"
        else:
            full_prompt = prompt
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Start with BOS + prompt + SEP
        initial_ids = [self.bos_id] + prompt_ids + [self.sep_id]
        
        # Initialize beams: (score, token_ids, is_finished)
        # Score is negative log probability (lower is better)
        beams = [(0.0, initial_ids, False)]
        
        # Completed sequences
        completed = []
        
        for step in range(max_new_tokens):
            all_candidates = []
            
            for score, token_ids, is_finished in beams:
                if is_finished:
                    completed.append((score, token_ids))
                    continue
                
                # Get logits for next token
                logits = self._get_logits(np.array(token_ids))
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits.copy(), token_ids, repetition_penalty)
                
                # Convert to log probabilities
                logits_max = np.max(logits)
                log_probs = logits - logits_max - np.log(np.sum(np.exp(logits - logits_max)))
                
                # Get top-k candidates
                top_k_indices = np.argsort(log_probs)[-beam_width * 2:][::-1]
                
                for token_id in top_k_indices:
                    new_score = score - log_probs[token_id]  # Accumulate negative log prob
                    new_token_ids = token_ids + [int(token_id)]
                    is_finished = (token_id == self.eos_id)
                    
                    # Apply length penalty
                    response_len = len(new_token_ids) - len(initial_ids)
                    length_factor = ((5 + response_len) / 6) ** length_penalty
                    adjusted_score = new_score / length_factor
                    
                    all_candidates.append((adjusted_score, new_score, new_token_ids, is_finished))
            
            # If no candidates, we're done
            if not all_candidates:
                break
            
            # Sort by adjusted score and keep top beam_width
            all_candidates.sort(key=lambda x: x[0])
            beams = [(c[1], c[2], c[3]) for c in all_candidates[:beam_width]]
            
            # Check if all beams are finished
            if all(b[2] for b in beams):
                completed.extend([(b[0], b[1]) for b in beams])
                break
        
        # Add remaining beams to completed
        completed.extend([(b[0], b[1]) for b in beams if not b[2]])
        
        # Sort by score and get top sequences
        completed.sort(key=lambda x: x[0])
        
        results = []
        for score, token_ids in completed[:num_return_sequences]:
            # Decode response (everything after SEP)
            if self.sep_id in token_ids:
                sep_pos = token_ids.index(self.sep_id)
                response_ids = token_ids[sep_pos + 1:]
            else:
                response_ids = token_ids
            
            # Filter out special tokens
            response_ids = [t for t in response_ids if t not in 
                           [self.bos_id, self.eos_id, self.sep_id, self.pad_id]]
            
            results.append(self.tokenizer.decode(response_ids))
        
        # Update conversation history if using context
        if use_context and results:
            self.conversation_history.add_turn(prompt, results[0])
        
        return results if results else [""]
    
    def clear_conversation_history(self):
        """Clear the conversation history for a fresh start."""
        self.conversation_history.clear()


def demonstrate_generation_strategies():
    """Demonstrate different generation strategies."""
    print("\n" + "=" * 60)
    print("GENERATION STRATEGIES EXPLAINED")
    print("=" * 60)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  GREEDY DECODING                                                        │
    │  ─────────────────                                                      │
    │  Always pick the highest probability token.                              │
    │                                                                         │
    │  Pros:  Fast, deterministic                                              │
    │  Cons: Can be repetitive, misses diverse outputs                        │
    │                                                                         │
    │  Example:                                                                │
    │  Logits: [0.1, 0.7, 0.15, 0.05]                                        │
    │  Always picks index 1 (prob 0.7)                                        │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TEMPERATURE SAMPLING                                                   │
    │  ────────────────────                                                   │
    │  Divide logits by temperature before softmax.                           │
    │                                                                         │
    │  Temperature < 1.0: More confident (peaky distribution)                 │
    │  Temperature = 1.0: Original distribution                               │
    │  Temperature > 1.0: More random (flatter distribution)                  │
    │                                                                         │
    │  Example with logits [2.0, 1.0, 0.5]:                                  │
    │  T=0.5: probs = [0.84, 0.11, 0.05]  (very confident)                  │
    │  T=1.0: probs = [0.59, 0.24, 0.17]  (original)                        │
    │  T=2.0: probs = [0.43, 0.32, 0.25]  (more uniform)                    │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TOP-K SAMPLING                                                         │
    │  ──────────────                                                         │
    │  Only consider the top k most likely tokens.                           │
    │                                                                         │
    │  Example with k=3:                                                      │
    │  Original probs: [0.4, 0.3, 0.15, 0.1, 0.05]                          │
    │  After top-k:     [0.47, 0.35, 0.18, 0, 0]  (renormalized)             │
    │                                                                         │
    │  Pros: Prevents very unlikely tokens                                    │
    │  Cons: Fixed k might cut off good tokens or include bad ones           │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  TOP-P (NUCLEUS) SAMPLING                                               │
    │  ────────────────────────                                               │
    │  Sample from smallest set of tokens whose cumulative probability ≥ p    │
    │                                                                         │
    │  Example with p=0.9:                                                    │
    │  Sorted probs: [0.5, 0.25, 0.15, 0.07, 0.03]                          │
    │  Cumulative:    [0.5, 0.75, 0.90, 0.97, 1.0]                           │
    │  Include tokens until cumulative ≥ 0.9 → first 3 tokens               │
    │                                                                         │
    │  Pros: Adapts to distribution (more tokens when uncertain)              │
    │  Cons:  Slightly more complex                                            │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  RECOMMENDED SETTINGS FOR DIFFERENT USE CASES                           │
    │  ─────────────────────────────────────────────────────────────────────  │
    │                                                                         │
    │  Factual/Consistent:   temperature=0.3, top_p=0.9                      │
    │  Balanced:             temperature=0.7, top_p=0.9                      │
    │  Creative:             temperature=1.0, top_p=0.95                     │
    │  Very Creative:        temperature=1.2, top_k=50                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """)


# ==================== INTERACTIVE CHAT ====================

def interactive_chat(generator:  TextGenerator):
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("TINYGREET INTERACTIVE CHAT")
    print("=" * 60)
    print("\nType your greeting and I'll respond!")
    print("Commands:  'quit' to exit, 'settings' to change generation settings")
    print("-" * 60)
    
    temperature = 0.7
    top_p = 0.9
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input: 
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!  👋")
            break
        
        if user_input. lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  Temperature: {temperature}")
            print(f"  Top-p:  {top_p}")
            try:
                temp_input = input("New temperature (0.1-2.0, or Enter to keep): ").strip()
                if temp_input: 
                    temperature = float(temp_input)
                
                top_p_input = input("New top_p (0.1-1.0, or Enter to keep): ").strip()
                if top_p_input:
                    top_p = float(top_p_input)
                
                print(f"Settings updated:  temperature={temperature}, top_p={top_p}")
            except ValueError:
                print("Invalid input, keeping current settings.")
            continue
        
        # Generate response
        response = generator.chat(
            prompt=user_input,
            temperature=temperature,
            top_p=top_p
        )
        
        print(f"Bot: {response}")


# ==================== TEST ====================

def test_generation():
    """Test text generation (with mock model)."""
    print("\n" + "=" * 60)
    print("TESTING TEXT GENERATION")
    print("=" * 60)
    
    # Create a mock model and tokenizer for testing
    class MockConfig:
        vocab_size = 100
        pad_token_id = 0
        bos_token_id = 2
        eos_token_id = 3
        sep_token_id = 4
    
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
        
        def __call__(self, token_ids, training=False):
            # Return random logits
            seq_len = len(token_ids)
            logits = np.random.randn(seq_len, self. config.vocab_size)
            # Make EOS more likely after some tokens
            if seq_len > 10:
                logits[:, self.config.eos_token_id] += 2.0
            return Tensor(logits)
    
    class MockTokenizer: 
        def __init__(self):
            self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, '<SEP>': 4}
        
        def encode(self, text, add_special_tokens=True):
            # Simple mock encoding
            tokens = [ord(c) % 95 + 5 for c in text[: 20]]
            if add_special_tokens:
                tokens = [2] + tokens + [3]
            return tokens
        
        def decode(self, token_ids):
            # Simple mock decoding
            chars = [chr((t - 5) % 95 + 32) if 5 <= t < 100 else '?' for t in token_ids]
            return ''.join(chars)
    
    model = MockModel()
    tokenizer = MockTokenizer()
    generator = TextGenerator(model, tokenizer, max_length=20)
    
    print("\n--- Greedy Generation ---")
    result = generator.generate_greedy("Hello")
    print(f"Prompt: 'Hello'")
    print(f"Response: '{result}'")
    
    print("\n--- Temperature Sampling ---")
    for temp in [0.5, 1.0, 1.5]:
        result = generator.generate("Hello", temperature=temp)[0]
        print(f"Temperature {temp}: '{result}'")
    
    print("\n--- Top-k Sampling ---")
    result = generator.generate("Hello", top_k=10)[0]
    print(f"Top-k=10: '{result}'")
    
    print("\n--- Top-p Sampling ---")
    result = generator.generate("Hello", top_p=0.9)[0]
    print(f"Top-p=0.9: '{result}'")
    
    print("\n--- Multiple Sequences ---")
    results = generator.generate("Hello", num_return_sequences=3)
    for i, r in enumerate(results):
        print(f"  Sequence {i+1}: '{r}'")
    
    print("\n✅ Generation tests passed!")
    
    # Show explanation
    demonstrate_generation_strategies()


if __name__ == "__main__":
    test_generation()