import numpy as np
from torch.utils.data import Dataset
import jsonlines
import vocab_gen

class YelpDataset(Dataset):
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file

        self.reviews = []

        with jsonlines.open(self.jsonl_file) as reader:
            for obj in reader.iter(type=dict, skip_invalid=True):
                rating = int(obj["stars"])
                review = obj["text"]

                self.reviews.append({"input": review, "label": rating})

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx]

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
        return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    yelp = YelpDataset("yelp_review_training_dataset.jsonl")

    