"""
Master Dataset Builder for TinyGreet

Combines: 
1. Hand-crafted seed data (high quality, fewer samples)
2. Systematically generated data (more samples, template-based)
3. Augmented variations

This is the final step before tokenization.
"""

import json
import os
import random
from typing import List, Dict
from collections import defaultdict

# Import our modules
from data_generator import GreetingDataGenerator


def load_json(filepath: str) -> List[Dict]: 
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deduplicate(data: List[Dict]) -> List[Dict]:
    """Remove duplicate input-output pairs."""
    seen = set()
    unique = []
    
    for sample in data:
        key = (sample["input"]. lower().strip(), sample["output"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(sample)
    
    return unique


def stratified_split(
    data: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42
) -> tuple: 
    """
    Split data while maintaining category distribution.
    """
    random.seed(seed)
    
    # Group by category and formality
    groups = defaultdict(list)
    for sample in data:
        key = (sample["category"], sample["formality"])
        groups[key].append(sample)
    
    train, val, test = [], [], []
    
    for key, samples in groups. items():
        random.shuffle(samples)
        n = len(samples)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(samples[:n_train])
        val.extend(samples[n_train:n_train + n_val])
        test.extend(samples[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def build_dataset():
    """Main dataset building pipeline."""
    print("=" * 70)
    print("TinyGreet Dataset Builder")
    print("=" * 70)
    
    # Step 1: Generate systematic data
    print("\n📝 Step 1: Generating systematic data...")
    generator = GreetingDataGenerator(seed=42)
    generated = generator.generate_all()
    print(f"   Generated:  {len(generated)} samples")
    
    # Step 2: Load seed data if exists
    seed_data = []
    seed_path = "splits/train.json"
    if os.path. exists(seed_path):
        print("\n📂 Step 2: Loading seed data...")
        seed_data = load_json(seed_path)
        # Also load val and test
        if os.path.exists("splits/val.json"):
            seed_data.extend(load_json("splits/val.json"))
        if os. path.exists("splits/test.json"):
            seed_data.extend(load_json("splits/test.json"))
        print(f"   Loaded:  {len(seed_data)} seed samples")
    else:
        print("\n📂 Step 2: No seed data found, using generated only")
    
    # Step 3: Combine and deduplicate
    print("\n🔄 Step 3: Combining and deduplicating...")
    all_data = seed_data + generated
    print(f"   Before dedup: {len(all_data)}")
    
    all_data = deduplicate(all_data)
    print(f"   After dedup:   {len(all_data)}")
    
    # Step 4: Add final IDs
    print("\n🏷️  Step 4: Assigning IDs...")
    for i, sample in enumerate(all_data):
        sample["id"] = f"tg_{i:05d}"
    
    # Step 5: Split
    print("\n✂️  Step 5: Splitting data...")
    train, val, test = stratified_split(all_data)
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    # Step 6: Save
    print("\n💾 Step 6: Saving datasets...")
    os.makedirs("final", exist_ok=True)
    
    save_json(train, "final/train. json")
    save_json(val, "final/val. json")
    save_json(test, "final/test. json")
    save_json(all_data, "final/all_data. json")
    
    # Also save a simple format for tokenizer training
    all_texts = []
    for sample in all_data:
        # Format: input ||| output
        all_texts.append(f"{sample['input']} ||| {sample['output']}")
    
    with open("final/corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n". join(all_texts))
    print(f"   Saved corpus. txt with {len(all_texts)} lines")
    
    # Step 7: Print final statistics
    print("\n" + "=" * 70)
    print("FINAL DATASET STATISTICS")
    print("=" * 70)
    
    stats = defaultdict(lambda: defaultdict(int))
    for sample in all_data:
        stats["category"][sample["category"]] += 1
        stats["formality"][sample["formality"]] += 1
        stats["subcategory"][sample["subcategory"]] += 1
    
    print(f"\n📊 Total samples: {len(all_data)}")
    
    print("\n📁 By Category:")
    for cat, count in sorted(stats["category"].items()):
        pct = count / len(all_data) * 100
        print(f"   {cat:15} {count:5} ({pct: 5.1f}%)")
    
    print("\n🎭 By Formality:")
    for form, count in sorted(stats["formality"].items()):
        pct = count / len(all_data) * 100
        print(f"   {form:<15} {count:5} ({pct:5.1f}%)")
    
    print("\n📂 By Subcategory:")
    for subcat, count in sorted(stats["subcategory"].items()):
        print(f"   {subcat:20} {count:4}")
    
    # Vocabulary statistics (for tokenizer planning)
    all_text = " ".join([s["input"] + " " + s["output"] for s in all_data])
    chars = set(all_text)
    words = set(all_text. lower().split())
    
    print(f"\n📝 Vocabulary Preview:")
    print(f"   Unique characters: {len(chars)}")
    print(f"   Unique words: {len(words)}")
    print(f"   Total characters: {len(all_text)}")
    
    print("\n" + "=" * 70)
    print("✅ Dataset build complete!")
    print("=" * 70)
    print("\nFiles created:")
    print("   📄 final/train.json")
    print("   📄 final/val. json")
    print("   📄 final/test.json")
    print("   📄 final/all_data.json")
    print("   📄 final/corpus.txt (for tokenizer training)")
    print("\n🚀 Ready for Phase 1. 2:  Tokenizer!")


if __name__ == "__main__": 
    build_dataset()