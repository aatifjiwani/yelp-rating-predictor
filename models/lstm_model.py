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

    def test(self, sequenced_review):
        return self.model.predict_classes(np.array(sequenced_review).reshape(1, -1))[0]+1

    def test_challenge_set(self, num, model_path):
        test_yelp = YelpDataset("../datasets/yelp_challenge_" + str(num) + "_with_answers.jsonl")
        x_test, y_test = test_yelp.make_datasets(t, 1000, "x_test_"+ str(num) +".txt", "y_test_"+ str(num) +".txt")
        self.load_model(model_path)
        mae = 0.0
        acc = 0.0
        for i in range(len(x_test)):
            pred = self.test(x_test[i])
            mae += abs(np.argwhere(y_test[i] == 1)[0][0] + 1 - pred)
            if np.argwhere(y_test[i] == 1)[0][0] + 1 == pred:
                acc += 1
        mae = mae / len(x_test)
        acc = acc / len(x_test)
        return mae, acc

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
    # x_train, y_train = yelp.make_datasets(t, 1000, "x_train.txt", "y_train.txt")
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
    mae_5, acc_5 = l.test_challenge_set(5, "model_lstm.model")
    mae_6, acc_6 = l.test_challenge_set(6, "model_lstm.model")
    print("mae_5: " + str(mae_5))
    print("acc_5: " + str(acc_5))
    print("mae_6: " + str(mae_6))
    print("acc_6: " + str(acc_6))
    # print(l.test("I went to this campus for 1 semester. I was in Business - Information Systems.\n\nThe campus is okay. The food choices are bismal.\n\nThe building is laid with the cafeteria on the bottom level, and then classes on the 2nd, 3rd, and 4th with each faculty basically having their own floor.\n\nTHe campus is pretty enough, but have fun getting the elevator around class start times...you're better to just stair it. \n\n\nIt's Seneca College after all."))



