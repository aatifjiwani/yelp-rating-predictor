from simpletransformers.classification import ClassificationModel
import sys

class PretrainedBert():
    def __init__(self):
        # CHANGE CUDA WHEN NEEDED
        # experiment w these args
        # change bert to other transformer
        # change pretrain to other pretrain
        self.model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, num_labels=5, args={'max_seq_length':128,'save_steps':10000, 'fp16':False, 'logging_steps': 1, 'train_batch_size':16, 'num_train_epochs':1})

    def run(self, train_df):
        self.model.train_model(train_df)

    def eval(self, eval_df):
        return self.model.predict(eval_df)


if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from YelpDataset import YelpDataset

    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl", using_pandas=True)
    b = PretrainedBert()
    b.run(yelp.train_df)
    yelp.make_eval_pandas(5)
    predictions, raw_outputs = b.eval(yelp.eval_df)
    print(predictions)
    print(raw_outputs)