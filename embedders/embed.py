import numpy as np

class Embedding():

    def __init__(self, tokenizer, embed_dim):
        self.tokenizer = tokenizer
        self.embeddings_index = dict()
        self.embedding_matrix = np.zeros((len(tokenizer), embed_dim))

    def load_embedding(self, pretrained_embedding_path):
        f = open(pretrained_embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coeffs
        f.close()
        print('Loaded %s word vectors.' % len(self.embeddings_index))

    def embed(self):
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector
            else:
                print("Word not in pretrained: " + word)
        print("Length embedding matrix: " + len(embedding_matrix))
