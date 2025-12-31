"""
Data Processing Pipeline for TinyGreet

This script: 
1. Loads the seed dataset
2. Normalizes and cleans the data
3. Creates variations for data augmentation
4. Splits into train/validation/test sets
5. Saves in our standard format
"""

import json
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import os

# We'll import our seed data
from raw.seed_dataset import SEED_DATA


class DataProcessor:
    """Process and augment the greeting/farewell dataset."""
    
    def __init__(self, seed:  int = 42):
        self.random = random.Random(seed)
        self.processed_data:  List[Dict] = []
        self.stats = defaultdict(int)
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving important features.
        
        Steps:
        1. Strip leading/trailing whitespace
        2. Normalize multiple spaces to single space
        3. Keep original capitalization (important for greetings)
        4. Keep punctuation (important for tone)
        """
        # Strip whitespace
        text = text.strip()
        
        # Normalize multiple spaces
        text = re. sub(r'\s+', ' ', text)
        
        return text
    
    def create_id(self, category: str, index: int) -> str:
        """Create a unique ID for each sample."""
        return f"{category}_{index: 04d}"
    
    def augment_sample(self, sample:  Dict, category: str) -> List[Dict]:
        """
        Create variations of a sample for data augmentation. 
        
        Techniques:
        1. Case variations (for casual greetings)
        2. Punctuation variations
        3. Contraction variations
        """
        variations = [sample. copy()]  # Original
        
        input_text = sample['input']
        output_text = sample['output']
        
        # Only augment casual greetings
        if sample. get('formality') == 'casual': 
            
            # Lowercase variation
            if input_text[0].isupper():
                var = sample.copy()
                var['input'] = input_text[0].lower() + input_text[1:]
                var['augmented'] = True
                variations.append(var)
            
            # ALL CAPS variation (for enthusiastic)
            if sample.get('mood') == 'enthusiastic':
                var = sample.copy()
                var['input'] = input_text. upper()
                var['augmented'] = True
                variations.append(var)
            
            # Add/remove exclamation marks
            if input_text. endswith('!'):
                var = sample.copy()
                var['input'] = input_text. rstrip('! ') + '.'
                var['augmented'] = True
                variations.append(var)
            elif input_text.endswith('. '):
                var = sample.copy()
                var['input'] = input_text. rstrip('.') + '!'
                var['augmented'] = True
                variations. append(var)
        
        return variations
    
    def infer_category(self, category_key: str) -> Tuple[str, str]:
        """Infer main category and subcategory from the key."""
        if 'morning' in category_key or 'afternoon' in category_key or \
           'evening' in category_key or 'night' in category_key: 
            return ('greeting', 'time_based')
        elif 'farewell' in category_key:
            return ('farewell', 'general')
        elif 'how_are_you' in category_key: 
            return ('greeting', 'how_are_you')
        elif 'meeting' in category_key:
            return ('greeting', 'first_meeting')
        elif 'returning' in category_key:
            return ('greeting', 'returning')
        elif 'thanks' in category_key: 
            return ('response', 'thanks')
        elif 'apolog' in category_key:
            return ('response', 'apology')
        else:
            return ('greeting', 'general')
    
    def process_seed_data(self, augment: bool = True) -> List[Dict]:
        """
        Process all seed data into standardized format.
        """
        processed = []
        global_index = 0
        
        for category_key, samples in SEED_DATA.items():
            main_category, subcategory = self.infer_category(category_key)
            
            for sample in samples:
                # Normalize texts
                input_text = self. normalize_text(sample['input'])
                output_text = self.normalize_text(sample['output'])
                
                # Create base processed sample
                base_sample = {
                    'id': self.create_id(category_key, global_index),
                    'input': input_text,
                    'output': output_text,
                    'category': main_category,
                    'subcategory': subcategory,
                    'formality': sample.get('formality', 'neutral'),
                    'time_of_day':  sample.get('time_of_day', 'any'),
                    'mood':  sample.get('mood', 'neutral'),
                    'augmented': False
                }
                
                # Get augmentations
                if augment:
                    variations = self.augment_sample(base_sample, category_key)
                    for var in variations: 
                        processed.append(var)
                        self.stats[main_category] += 1
                else:
                    processed.append(base_sample)
                    self.stats[main_category] += 1
                
                global_index += 1
        
        self.processed_data = processed
        return processed
    
    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio:  float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/validation/test sets.
        
        Uses stratified splitting to maintain category distribution.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Group by category for stratified split
        by_category = defaultdict(list)
        for sample in self. processed_data: 
            key = (sample['category'], sample['formality'])
            by_category[key].append(sample)
        
        train, val, test = [], [], []
        
        for key, samples in by_category.items():
            self.random.shuffle(samples)
            n = len(samples)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train.extend(samples[:train_end])
            val.extend(samples[train_end:val_end])
            test.extend(samples[val_end:])
        
        # Shuffle each split
        self.random.shuffle(train)
        self.random.shuffle(val)
        self.random.shuffle(test)
        
        return train, val, test
    
    def save_splits(
        self,
        train:  List[Dict],
        val: List[Dict],
        test: List[Dict],
        output_dir: str = 'splits'
    ):
        """Save the data splits to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in [('train', train), ('val', val), ('test', test)]:
            filepath = os.path. join(output_dir, f'{name}.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(data)} samples to {filepath}")
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        
        print(f"\nTotal samples: {len(self.processed_data)}")
        
        print("\nBy category:")
        for cat, count in sorted(self.stats. items()):
            print(f"  {cat}: {count}")
        
        # Count augmented vs original
        augmented = sum(1 for s in self.processed_data if s. get('augmented', False))
        original = len(self.processed_data) - augmented
        print(f"\nOriginal samples: {original}")
        print(f"Augmented samples:  {augmented}")
        
        # Formality distribution
        print("\nBy formality:")
        formality_counts = defaultdict(int)
        for s in self.processed_data:
            formality_counts[s['formality']] += 1
        for form, count in sorted(formality_counts. items()):
            print(f"  {form}: {count}")
        
        # Calculate average lengths
        input_lengths = [len(s['input']) for s in self. processed_data]
        output_lengths = [len(s['output']) for s in self.processed_data]
        
        print(f"\nAverage input length: {sum(input_lengths)/len(input_lengths):.1f} chars")
        print(f"Average output length: {sum(output_lengths)/len(output_lengths):.1f} chars")
        print(f"Max input length: {max(input_lengths)} chars")
        print(f"Max output length: {max(output_lengths)} chars")


def main():
    """Main processing pipeline."""
    print("TinyGreet Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = DataProcessor(seed=42)
    
    # Process seed data
    print("\n1. Processing seed data...")
    processed = processor.process_seed_data(augment=True)
    print(f"   Processed {len(processed)} samples")
    
    # Print statistics
    processor. print_statistics()
    
    # Split data
    print("\n2. Splitting data...")
    train, val, test = processor.split_data(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save splits
    print("\n3. Saving splits...")
    processor.save_splits(train, val, test)
    
    print("\n" + "=" * 50)
    print("Processing complete!")


if __name__ == "__main__": 
    main()