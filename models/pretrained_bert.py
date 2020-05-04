from simpletransformers.classification import ClassificationModel
import sys

class PretrainedBert():
    def __init__(self):
        # CHANGE CUDA WHEN NEEDED
        self.model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, num_labels=5, args={'max_seq_length':1000, 'train_batch_size':128, 'num_train_epochs':3})

    def run(self, train_df):
        # experiment w these args
        self.model.train_model(train_df)

    def eval(self, eval_df):
        return self.model.eval_model(eval_df)


if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from YelpDataset import YelpDataset

    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl", using_pandas=True)
    b = PretrainedBert()
    b.run(yelp.train_df)
    yelp.make_eval_pandas(5)
    predictions, raw_outputs = b.predict(yelp.eval_df)
    print(predictions)
    print(raw_outputs)