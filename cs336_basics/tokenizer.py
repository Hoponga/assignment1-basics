import regex as re
import os
from typing import BinaryIO
import collections

from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# usage: re.findall(PAT, "some text that i'll pre-tokenize")


def pretokenize_chunk(chunk: str): 
    pre_tokens = re.finditer(PAT, chunk)

    token_counts = collections.defaultdict(int) 

    for token in pre_tokens: 
        enc_token = token.group().encode('utf-8')
        token_counts[enc_token] += 1

    #print(token_counts)
    return token_counts 


# reference: https://arxiv.org/pdf/1508.07909 algorithm 1
def get_stats(vocab: dict): 
    pairs = collections.defaultdict(int)

    for word, count in vocab: 
        symbols = word.split() 
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += count
        
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

if __name__ == "__main__": 

    DATA_FILE = "../../data/TinyStoriesV2-GPT4-valid.txt" 
    NUM_MERGE_STEPS = 10

    with open(DATA_FILE, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            print("PRETOKENIZING CHUNK")
            pre_tokens = pretokenize_chunk(chunk) 
            new_vocab = pre_tokens 

            for i in range(NUM_MERGE_STEPS): 


                pair_stats = get_stats(new_vocab)

                best_pair = max(pair_stats, key = pair_stats.get)

                new_vocab = merge_vocab(best_pair, new_vocab)
            print(f"Final vocab: {new_vocab}")
            print(f"Final # of tokens: {len(new_vocab.keys())}")










