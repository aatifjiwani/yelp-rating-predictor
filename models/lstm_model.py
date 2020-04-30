from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

class LSTM_Model():

    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def run(self):
        model = Sequential()
        # ADD INPUT SIZE
        model.add(Embedding(len(self.embedding_matrix),200,    INPUT SIZE    , weights=self.embedding_matrix, trainable=False))
        model.add(Dropout(0.2))
        model.add(Conv1D(64,5,activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(200))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        # ADD THESE
        model.fit(TRAINING REVIEW DATA, TRAINING RATING DATA, VALIDATION SPLIT, EPOCHS)

if __name__=="__main__":
    sys.path.insert(1, '../datasets/')
    from vocab_gen import Tokenizer
    from YelpDataset import YelpDataset
    t = Tokenizer("yo", "vocabulary.txt")

    sys.path.insert(1, '../embedders/')
    from embedded import Embedding

    em = Embedding(t)
    em.load_embedding("embedded.txt")
    vocab_embedded = em.embed(200)

    l = LSTM_Model(vocab_embedded)
    l.run()



