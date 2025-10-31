"""
Download real prompt datasets for MBB benchmarking.

Downloads GSM8K math problems and creates three workload mixes:
- short_heavy: Short prompts (single sentence)
- mixed: Mix of short, medium, long
- long_tail: Mostly short with some very long
"""

import os
import json
import random
from typing import List, Dict


def download_gsm8k_prompts() -> List[str]:
    """Download GSM8K dataset prompts.
    
    Returns:
        List of math problem prompts
    """
    # For now, use sample prompts
    # In production, download from: https://github.com/openai/grade-school-math
    
    sample_prompts = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "What is 2 + 2?",
        "Calculate 15 * 8",
        "What is the square root of 144?",
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "A store has 50 apples. They sell 23. How many are left?",
        "Convert 100 degrees Fahrenheit to Celsius.",
        "What is 1/4 + 1/2?",
        "A rectangle has length 10 and width 5. What is its area?",
        "Solve for x: 2x + 5 = 13",
        "What is 20% of 150?",
        # Medium prompts
        "A farmer has 3 fields. In the first field, he plants 120 rows of corn with 15 stalks per row. In the second field, he plants 80 rows with 20 stalks per row. In the third field, he plants 100 rows with 18 stalks per row. If each stalk produces 2 ears of corn, and he sells each ear for $0.25, how much money will he make from all three fields?",
        "Maria is planning a party. She needs to buy plates, cups, and napkins. Plates come in packs of 12 for $3.50, cups in packs of 16 for $4.00, and napkins in packs of 20 for $2.50. If she's expecting 45 guests and wants 2 of each item per guest, plus 10 extras of everything, how much will she spend in total?",
        "A library has 5 shelves. Each shelf has 8 sections. Each section can hold 25 books. Currently, the library is 60% full. How many more books can be added before the library is completely full?",
        # Long prompts
        "A company manufactures widgets. They have three production lines. Line A produces 120 widgets per hour and operates 8 hours per day. Line B produces 150 widgets per hour and operates 10 hours per day. Line C produces 100 widgets per hour and operates 12 hours per day. Each widget costs $2 in materials and $1.50 in labor to produce. The company sells widgets for $8 each. If they operate 5 days a week for 4 weeks, and their fixed costs (rent, utilities, etc.) are $5000 per week, what is their monthly profit? Additionally, if they want to increase profit by 20% next month, how many more widgets per day do they need to produce assuming they can sell all of them?",
        "A school district is planning transportation for field trips. They have 450 students across 3 schools. School A has 180 students, School B has 150 students, and School C has 120 students. They can rent two types of buses: small buses that hold 30 students and cost $150 per day, or large buses that hold 50 students and cost $220 per day. Each school needs at least 2 adult chaperones for every 15 students. The district has 40 available adult chaperones total. The field trip location is 45 miles away, and buses get 8 miles per gallon with diesel costing $3.50 per gallon. What is the minimum cost to transport all students, considering bus rental, fuel costs, and the constraint on chaperones? If the district has a budget of $5000, can they afford the trip?",
    ]
    
    return sample_prompts


def create_workload_mixes(prompts: List[str], size: int = 60) -> Dict[str, List[str]]:
    """Create three workload mixes from prompts.
    
    Args:
        prompts: List of all available prompts
        size: Number of prompts per mix (default 60, can be 1000 for larger benchmarks)
        
    Returns:
        Dictionary with 'short_heavy', 'mixed', 'long_tail' lists
    """
    # Sort by length
    sorted_prompts = sorted(prompts, key=len)
    
    n = len(sorted_prompts)
    third = n // 3
    
    short = sorted_prompts[:third]
    medium = sorted_prompts[third:2*third]
    long_prompts = sorted_prompts[2*third:]
    
    # Ensure we have enough prompts by repeating if needed
    while len(short) < size:
        short = short * 2
    while len(medium) < size:
        medium = medium * 2
    while len(long_prompts) < size:
        long_prompts = long_prompts * 2
    
    # Short-heavy: 80% short, 20% medium
    short_count = int(size * 0.8)
    medium_count = size - short_count
    short_heavy = random.sample(short, short_count) + random.sample(medium, medium_count)
    random.shuffle(short_heavy)
    
    # Mixed: 50% short, 30% medium, 20% long
    short_count = int(size * 0.5)
    medium_count = int(size * 0.3)
    long_count = size - short_count - medium_count
    mixed = (random.sample(short, short_count) + 
             random.sample(medium, medium_count) + 
             random.sample(long_prompts, long_count))
    random.shuffle(mixed)
    
    # Long-tail: 70% short, 20% medium, 10% long
    short_count = int(size * 0.7)
    medium_count = int(size * 0.2)
    long_count = size - short_count - medium_count
    long_tail = (random.sample(short, short_count) + 
                 random.sample(medium, medium_count) + 
                 random.sample(long_prompts, long_count))
    random.shuffle(long_tail)
    
    return {
        'short_heavy': short_heavy[:size],
        'mixed': mixed[:size],
        'long_tail': long_tail[:size],
        'mixed_1k': mixed[:min(1000, len(mixed) * 17)]  # Create 1k version for stability
    }


def save_prompts(mixes: Dict[str, List[str]], data_dir: str = "data"):
    """Save prompt mixes to text files.
    
    Args:
        mixes: Dictionary of prompt lists
        data_dir: Directory to save files
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for name, prompts in mixes.items():
        filepath = os.path.join(data_dir, f"{name}.txt")
        with open(filepath, 'w') as f:
            for prompt in prompts:
                # One prompt per line
                f.write(prompt.replace('\n', ' ') + '\n')
        print(f"Saved {len(prompts)} prompts to {filepath}")


def main():
    """Download and prepare datasets."""
    print("Downloading prompts...")
    prompts = download_gsm8k_prompts()
    
    print(f"\nGot {len(prompts)} prompts")
    print("Creating workload mixes...")
    
    random.seed(42)
    # Create both 60-prompt and 1k-prompt versions
    mixes_60 = create_workload_mixes(prompts, size=60)
    mixes_1k = create_workload_mixes(prompts, size=1000)
    
    # Combine for saving
    all_mixes = {**mixes_60, **{f"{k}_1k": v for k, v in mixes_1k.items()}}
    
    print("\nSaving to files...")
    save_prompts(all_mixes)
    
    # Print stats
    print("\nWorkload Mix Statistics:")
    for name, prompts_list in all_mixes.items():
        avg_len = sum(len(p) for p in prompts_list) / len(prompts_list)
        min_len = min(len(p) for p in prompts_list)
        max_len = max(len(p) for p in prompts_list)
        print(f"\n{name}:")
        print(f"  Count: {len(prompts_list)}")
        print(f"  Avg length: {avg_len:.1f} chars")
        print(f"  Range: {min_len}-{max_len} chars")


if __name__ == "__main__":
    main()

