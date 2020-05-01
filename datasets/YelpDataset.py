import numpy as np
from torch.utils.data import Dataset
import jsonlines
import json
import vocab_gen
import random
from tqdm import tqdm

from vocab_gen import *

class YelpDataset(Dataset):
    def __init__(self, jsonl_file:str, tokenizer:Tokenizer=None, max_len:int = 50):
        self.jsonl_file = jsonl_file

        self.reviews = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with jsonlines.open(self.jsonl_file) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                # rating = obj["stars"]
                # review = obj["text"]

                self.reviews.append({"input": obj["input"], "label": obj["label"]})

        print("dataset loaded...")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        assert self.tokenizer is not None, "tokenizer must be passed in during instantiation"

        sample =  self.reviews[idx]
        review, stars = sample["input"], int(sample["label"])

        review = self.tokenizer.tokenize2Index(review)[:self.max_len]
        if (len(review) < self.max_len):
            review += [PAD_TOKEN]*(self.max_len-len(review))

        return {"input": np.array(review), "label": np.array(stars)}


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
            rating_vector[self.reviews[i]["label"]-1] = 1
            sequenced_review = tokenizer.tokenize2Index(self.reviews[i]["input"])
            if len(sequenced_review) > max_length:
                sequenced_review = sequenced_review[:max_length]
            elif len(sequenced_review) < max_length:
                sequenced_review += [vocab_gen.PAD_TOKEN]*(max_length-len(sequenced_review))
            sequenced_review = [int(x) for x in sequenced_review]
            if i <= int(.8*num_reviews):
                x_train.append(sequenced_review)
                y_train.append(rating_vector)
            else:
                x_val.append(sequenced_review)
                y_val.append(rating_vector)

        np.savetxt('x_train.txt', x_train, fmt ='%4d')
        np.savetxt('y_train.txt', y_train, fmt='%4d')
        np.savetxt('x_val.txt', x_val, fmt='%4d')
        np.savetxt('y_val.txt', y_val, fmt='%4d')
        return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


if __name__ == "__main__":
    # yelp = YelpDataset("yelp_review_training_dataset.jsonl")
    # print(len(yelp))
    # print(yelp[len(yelp) - 1])
    # yelp.split_dataset(0.8, "yelp_training.jsonl", "yelp_validation.jsonl")

    # yelp_train = YelpDataset("yelp_training.jsonl")

    # tokenizer = Tokenizer("global", "vocabulary.txt")
    # yelp_val = YelpDataset("yelp_validation.jsonl", tokenizer=tokenizer, max_len=100)
    # print(yelp_val.reviews[10])
    # print(yelp_val[10])

    # print(len(yelp_train))
    # print(len(yelp_val))
    # print(yelp_val[len(yelp_val) - 1]
    pass

