import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

class Embedding():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.embeddings_index = dict()
        self.embedding_matrix = None

    def load_embedding(self, pretrained_embedding_path):
        f = open(pretrained_embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coeffs
        f.close()
        print('Loaded %s word vectors.' % len(self.embeddings_index))

    # USE THIS FOR LSTM EMBEDDING WEIGHT
    # Gets embedded vector for each word in word2Index
    def embed(self, embed_dim):
        self.embedding_matrix = np.zeros((len(self.tokenizer), embed_dim))
        for word, index in self.tokenizer.word2Index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector
            else:
                print("Word not in pretrained: " + word)
        print("Length embedding matrix: " + len(embedding_matrix))

    # reviews should be a list of reviews
    def load_ELMO(self, reviews):
        url = "https://tfhub.dev/google/elmo/2"
        embed = hub.Module(url)
        embeddings = embed(
            reviews,
            signature="default",
            as_dict=True)["default"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            self.embeddings_matrix = sess.run(embeddings)
