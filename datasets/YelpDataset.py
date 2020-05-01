import numpy as np
from torch.utils.data import Dataset
import jsonlines
import json
import random
from tqdm import tqdm

try:
    from vocab_gen import *
except ImportError:
    from datasets.vocab_gen import *

class YelpDataset(Dataset):
    def __init__(self, jsonl_file:str, tokenizer:Tokenizer=None, max_len:int = 50, is_from_partition=False, should_stem=True):
        self.jsonl_file = jsonl_file

        self.reviews = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.should_stem = should_stem

        with jsonlines.open(self.jsonl_file) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                if is_from_partition:
                    self.reviews.append({"input": obj["input"], "label": obj["label"]})
                else:
                    rating = obj["stars"]
                    review = obj["text"]

                    self.reviews.append({"input": review, "label": rating})
                

        print("dataset loaded...")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        assert self.tokenizer is not None, "tokenizer must be passed in during instantiation"

        sample =  self.reviews[idx]
        review, stars = sample["input"], int(sample["label"])

        review = self.tokenizer.tokenize2Index(review, self.should_stem)[:self.max_len]
        if (len(review) < self.max_len):
            review += [PAD_TOKEN]*(self.max_len-len(review))

        return {"input": np.array(review), "label": np.array(stars - 1)}


    def split_dataset(self, training_partition: float, training_file: str, validation_file: str):
        assert training_partition > 0 and training_partition < 1, "Training partition must be a float between 0 and 1 non-exclusive"
        
        num_train_examples = int(training_partition * len(self.reviews))

        random.shuffle(self.reviews)

        training_partition = self.reviews[:num_train_examples]
        val_partition = self.reviews[num_train_examples:]

        with open(training_file, "w+") as train_f:
            for rev in tqdm(training_partition):
                json.dump(rev, train_f)
                train_f.write("\n")

        with open(validation_file, "w+") as val_f:
            for rev in tqdm(val_partition):
                json.dump(rev, val_f)
                val_f.write("\n")                

    def make_datasets(self, tokenizer, max_length):
        x_train, y_train, x_val, y_val = [],[],[],[]
        num_reviews = len(self.reviews)
        for i in range(num_reviews):
            rating_vector = [0,0,0,0,0]
            rating_vector[int(self.reviews[i]["label"])-1] = 1
            sequenced_review = tokenizer.tokenize2Index(self.reviews[i]["input"])
            if len(sequenced_review) > max_length:
                sequenced_review = sequenced_review[:max_length]
            elif len(sequenced_review) < max_length:
                sequenced_review += [PAD_TOKEN]*(max_length-len(sequenced_review))
            sequenced_review = [int(x) for x in sequenced_review]
            x_train.append(sequenced_review)
            y_train.append(rating_vector)

        np.savetxt('x_train.txt', x_train, fmt ='%4d')
        np.savetxt('y_train.txt', y_train, fmt='%4d')
        return np.asarray(x_train), np.asarray(y_train)

