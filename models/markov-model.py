import os
import torch
from collections import defaultdict
from itertools import count

def load_names(dataset_path):
    with open(dataset_path, 'r') as f:
        return f.read().splitlines()

def build_bigram_model(names):
    bi_pairs = defaultdict(lambda: defaultdict(int))
    chs = {ch for name in names for ch in name}
    chs.add('.')
    chs = sorted(chs)
    
    # Apply Laplace smoothing
    for ch1 in chs:
        for ch2 in chs:
            for ch3 in chs:
                bi_pairs[ch1 + ch2][ch3] += 1
    
    # Process names into bigrams
    for name in names:
        name = f'.{name}.'
        for ch1, ch2, ch3 in zip(name, name[1:], name[2:]):
            pair = ch1 + ch2
            bi_pairs[pair][ch3] += 1
    
    return bi_pairs, chs

def encode_bigram_output(bi_pairs, chs):
    bigram_id_gen = count()
    output_id_gen = count()

    bigram_ids = {}
    output_ids = {}
    
    for ch in chs:
        output_ids[next(output_id_gen)] = ch
    
    for bigram in bi_pairs.items():
        bigram_ids[next(bigram_id_gen)] = bigram[0]
    
    return bigram_ids, output_ids

def convert_counts_to_probabilities(bi_pairs):
    bi_pairs_prob = defaultdict(lambda: defaultdict(int))
    
    for bigram, outs in bi_pairs.items():
        total = sum(outs.values())
        for chr, count in outs.items():
            bi_pairs_prob[bigram][chr] = count / total
    
    return bi_pairs_prob

def generate_name(bi_pairs_prob, output_ids, chs):
    while True:
        word = ''
        chr = input("Enter first character of the name: ").lower()
        if chr not in chs:
            print("--------\nGoodbye!")
            break
        
        chr = '.' + chr[-1]
        word = f'{chr[-1]}'
        
        while True:
            chr_tensor = torch.tensor(list(bi_pairs_prob[chr].values()))
            sample = torch.multinomial(chr_tensor, 1, replacement=True)
            chr = chr[-1] + output_ids[sample.item()]
            if chr[-1] == '.':
                break
            word += chr[-1]
        
        print(f"Name starting with {word[0]}: {word}")

def main():
    dataset_path = input("Enter dataset location (e.g., ../data/names.txt): ").strip()
    if not os.path.exists(dataset_path):
        print("Error: File not found!")
        return
    
    names = load_names(dataset_path)
    bi_pairs, chs = build_bigram_model(names)
    bigram_ids, output_ids = encode_bigram_output(bi_pairs, chs)
    bi_pairs_prob = convert_counts_to_probabilities(bi_pairs)
    
    generate_name(bi_pairs_prob, output_ids, chs)

if __name__ == "__main__":
    main()
