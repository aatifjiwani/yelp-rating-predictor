import torch
from datasets.YelpDataset import YelpDataset
from embedders.skipgram import SkipGram
from datasets.vocab_gen import *
from typing import List
import numpy as np

epochs = 1

def train():
    yelp_dataset = YelpDataset('datasets/yelp_review_training_dataset.jsonl')
    tokenizer = Tokenizer("tokenizer", "datasets/vocabulary.txt")

    vocab_size = tokenizer.vocabSize()
    num_reviews = len(yelp_dataset)

    model = SkipGram(512, vocab_size)
    model.train()

    for epoch in range(epochs):
        batch = generateInput(yelp_dataset[0]["input"], tokenizer)
        inputs = batch[:, 0]
        targets = batch[:, 1]

        one_hot_inputs = one_hot_encode(inputs, vocab_size)

def generateInput(review: str, tokenizer: Tokenizer, window_size:int = 5) -> np.ndarray:
    tokenized_indices = tokenizer.tokenize2Index(review)

    center_context_pairs = []
    for center_pos in range(len(tokenized_indices)):
        center_idx = tokenized_indices[center_pos]
        for offset in range(-window_size, window_size+1):
            context_pos = center_pos + offset
            if (context_pos < 0 or context_pos >= len(tokenized_indices) or offset == 0):
                continue
                
            context_idx = tokenized_indices[context_pos]


            center_context_pairs.append([center_idx, context_idx])

    return np.array(center_context_pairs)

def one_hot_encode(batch:np.ndarray, vocab_size: int):
    one_hot_batch = torch.zeros(batch.shape[0], vocab_size)

    for word in range(batch.shape[0]):
        one_hot_batch[word][batch[word]] = 1

    return one_hot_batch

if __name__ == "__main__":
    train()