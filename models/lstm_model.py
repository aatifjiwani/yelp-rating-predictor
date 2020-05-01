from keras.models import Sequential
from keras.layers.embeddings import Embedding as Em
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
import matplotlib.pyplot as plt
import sys
import numpy as np

class LSTM_Model():

    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.model = None
        self.history = None

    def build(self):
        model = Sequential()
        # ADD INPUT SIZE
        model.add(Em(len(self.embedding_matrix), 200, input_length=1000, weights=[self.embedding_matrix], trainable=False))
        model.add(Bidirectional(LSTM(128, implementation=2)))
        model.add(Dropout(0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def run(self, x_t, y_t, x_v, y_v, save_path):
        self.history = self.model.fit(x_t,
                      y_t,
                      epochs=3,
                      batch_size=128,
                      validation_data=(x_v, y_v)
                       )
        print(self.history.history.keys())
        self.model.save(save_path)

    def plot_loss(self):

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        plt.savefig("LSTM_Loss.png", dpi=600)
        plt.show()

    def plot_acc(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.xticks(np.arange(0, 2, step=1))
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')

        plt.savefig("LSTM_Acc.png", dpi=300)
        plt.show()

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from vocab_gen import Tokenizer
    from YelpDataset import YelpDataset
    t = Tokenizer("yo", "../datasets/vocabulary.txt")
    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl")
    x_train, y_train, x_val, y_val = yelp.make_datasets(t, 1000)
    # with open('../models/x_train.txt', 'r') as f:
    #     x_train = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])
    # with open('../models/y_train.txt', 'r') as f:
    #     y_train = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])
    # with open('../models/x_val.txt', 'r') as f:
    #     x_val = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])
    # with open('../models/y_val.txt', 'r') as f:
    #     y_val = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])

    sys.path.insert(1, '../embedders/')
    from embed import Embedding

    em = Embedding(t)
    em.load_embedding("../embedders/embedded.txt")
    vocab_embedded = em.embed(200)

    l = LSTM_Model(vocab_embedded)
    l.build()

    l.run(x_train, y_train, x_val, y_val, "model_lstm.model")
    l.plot_acc()
    l.plot_loss()



