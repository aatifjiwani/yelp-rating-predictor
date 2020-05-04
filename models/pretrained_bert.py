from simpletransformers.classification import ClassificationModel
import sys

class PretrainedBert():
    def __init__(self):
        self.model = ClassificationModel('bert', 'bert-base-uncased', num_labels=5, args={'max_seq_length':1000, 'train_batch_size':128, 'num_train_epochs':3})

    def run(self, train_df):
        # experiment w these args
        self.model.train_model(train_df)


if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from YelpDataset import YelpDataset

    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl", using_pandas=True)
    b = PretrainedBert()
    b.run(yelp.train_df)
