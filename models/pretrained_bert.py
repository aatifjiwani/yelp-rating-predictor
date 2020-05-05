from simpletransformers.classification import ClassificationModel
import sys

class PretrainedBert():
    def __init__(self):
        # CHANGE CUDA WHEN NEEDED
        # experiment w these args
        # change bert to other transformer
        # change pretrain to other pretrain
        self.model = ClassificationModel('roberta', 'roberta-base', use_cuda=False, num_labels=5, args={'max_seq_length':128,'save_steps':10000, 'fp16':False, 'logging_steps': 1, 'train_batch_size':16, 'num_train_epochs':1})

    def run(self, train_df):
        self.model.train_model(train_df)

    def eval(self, eval_df):
        return self.model.predict(eval_df)

    def load(self, path):
        self.model = ClassificationModel('roberta', path, num_labels=5, use_cuda=False,args={'max_seq_length':128,'save_steps':10000,'fp16':False,'logging_steps': 1, 'train_batch_size':16, 'num_train_epochs':1})

    def test_challenge_set(self, labels_pred, labels_act):
        mae = 0.0
        acc = 0.0
        for i in range(len(labels_act)):
            mae += abs(labels_pred[i] - labels_act[i])
            if labels_pred[i] == labels_act[i]:
                acc += 1
        mae = mae / len(labels_act)
        acc = acc / len(labels_act)
        return mae, acc

if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from YelpDataset import YelpDataset

    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl", using_pandas=True)

    b = PretrainedBert()
    #b.run(yelp.train_df)
    b.load('outputs/')

    for i in range(5, 7):
        yelp.make_eval_pandas(i)
        predictions, raw_outputs = b.eval(yelp.eval_df["text"])
        mae, acc = b.test_challenge_set(predictions, yelp.eval_df["label"])
        print("mae_" + str(i) + ": " + str(mae))
        print("acc_" + str(i) + ": " + str(acc))

