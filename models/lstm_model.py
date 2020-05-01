from keras.models import Sequential
from keras.layers.embeddings import Embedding as Em
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

    def run(self, x_t, y_t, val_percentage):
        es = EarlyStopping(monitor='val_loss', min_delta=0, mode='min', verbose=1, patience=1)
        mc = ModelCheckpoint('best_lstm_model.model', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        self.history = self.model.fit(x_t,
                      y_t,
                      epochs=3,
                      batch_size=128,
                      validation_split=val_percentage,
                      callbacks=[es, mc]
                       )
        print(self.history.history.keys())

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

    def load_model(self, path):
        self.build()
        self.model.load_weights(path)

    def test(self, tokenizer, review, max_length):
        sequenced_review = tokenizer.tokenize2Index(review)
        if len(sequenced_review) > max_length:
            sequenced_review = sequenced_review[:max_length]
        elif len(sequenced_review) < max_length:
            sequenced_review += [PAD_TOKEN] * (max_length - len(sequenced_review))
        sequenced_review = [int(x) for x in sequenced_review]
        return self.model.predict_classes(np.array(sequenced_review.reshape(1, -1)))[0]+1


def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from vocab_gen import *
    from YelpDataset import YelpDataset
    t = Tokenizer("yo", "../datasets/vocabulary.txt")
    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl")
    # x_train, y_train = yelp.make_datasets(t, 1000)
    # with open('../models/x_train.txt', 'r') as f:
    #     x_train = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])
    # with open('../models/y_train.txt', 'r') as f:
    #     y_train = np.asarray([[int(idx.rstrip('\n')) for idx in line.split() if is_int(idx.rstrip('\n'))] for line in f])

    sys.path.insert(1, '../embedders/')
    from embed import Embedding

    em = Embedding(t)
    em.load_embedding("../embedders/embedded.txt")
    vocab_embedded = em.embed(200)

    l = LSTM_Model(vocab_embedded)
    # l.build()
    #
    # l.run(x_train, y_train, 0.2)
    # l.plot_acc()
    # l.plot_loss()

    l.load_model("model_lstm.model")
    print(l.test(t, "The food was okay. I think my pasta was a little bland, but the service was quick.", 1000))



