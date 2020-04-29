import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from gensim.models import Word2Vec
import sys
from nltk.stem import SnowballStemmer

class Embedding():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.embeddings_index = dict()

    def load_embedding(self, pretrained_embedding_path):
        stemmer = SnowballStemmer("english")
        f = open(pretrained_embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            word = stemmer.stem(word)
            coeffs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coeffs
        f.close()
        print('Loaded %s word vectors.' % len(self.embeddings_index))

    # USE THIS FOR LSTM EMBEDDING WEIGHT
    # Gets embedded vector for each word in word2Index
    def embed(self, embed_dim):
        embedding_matrix = np.zeros((len(self.tokenizer.word2Index), embed_dim))
        missed= []
        for word, index in self.tokenizer.word2Index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                missed.append(word)
        print("Length embedding matrix: " + str(len(embedding_matrix)))
        print("Words not embedded: " + str(len(missed)))
        return embedding_matrix

    # reviews should be a list of list of words in reviews
    def load_ELMO(self, module):
        # elmo = hub.Module("module/module_elmo2/", trainable=False)
        # review_lengths = []
        # for i in reviews:
        #     review_lengths.append(len(i))
        # embeddings = elmo(
        #     inputs={
        #         "tokens": reviews,
        #         "sequence_len": review_lengths
        #     },
        #     signature="tokens",
        #     as_dict=True)["elmo"]
        # with tf.Session() as sess:
        #     sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #     x = sess.run(embeddings)
        #     return x
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})

    # uses skip-gram
    def load_word2vec(self, reviews):
        return Word2Vec(reviews, min_count=1, size=200, workers=3, window=3, sg=1)

if __name__ == "__main__":
    sys.path.insert(1, '../datasets/')
    from vocab_gen import Tokenizer
    from YelpDataset import YelpDataset
    t = Tokenizer("yo", "../datasets/vocabulary.txt")
    em = Embedding(t)
    #switch between glove and conceptnet
    em.load_embedding("glove.twitter.27B/glove.twitter.27B.200d.txt")
    x = em.embed(200)
    print(x)

    # mkdir module/module_elmo2
    # curl -L "https://tfhub.dev/google/elmo/2?tf-hub-format=compressed" | tar -zxvC module/module_elmo2
    # em_elmo = Embedding(None)
    # x = em_elmo.load_ELMO("module/module_elmo2")
    # print(x(["yo name jeff"]))
    # print(x(["my jeff name yo"]))

    #em_w2c = Embedding(None)
    # x = em_w2c.load_word2vec([["my", "name", "jeff"],["your","name","jeff"], ["jeff", "yo", "man"],["why", "he", "not", "make", "my","taco"]])
    # print(x["jeff"])
    # print(x.most_similar("yo"))

    tokenized_reviews = []
    yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl")
    print("# reviews: " + str(len(yelp.reviews)))
    for r in yelp.reviews:
        y = em.tokenizer.tokenize(r["input"])
        print(y)
        tokenized_reviews.append(y)
    np.savetxt("reviews.txt",tokenized_reviews, fmt="%s")
    embedded = em.load_word2vec(tokenized_reviews)
    print(embedded.most_similar("nigger"))