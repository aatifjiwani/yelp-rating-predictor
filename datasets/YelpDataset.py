import numpy as np
from torch.utils.data import Dataset
import jsonlines

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
        return 

if __name__ == "__main__":
    yelp = YelpDataset("yelp_review_training_dataset.jsonl")

    