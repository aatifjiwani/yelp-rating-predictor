import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from gensim.models import Word2Vec
import sys
from nltk.stem import SnowballStemmer
import multiprocessing

class Embedding():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.embeddings_index = dict()

    def load_stem_embedding(self, pretrained_embedding_path):
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

    def load_embedding(self, pretrained_embedding_path):
        f = open(pretrained_embedding_path, encoding="utf-8")
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
        embedding_matrix = np.zeros((len(self.tokenizer.word2Index), embed_dim))
        missed= []
        for word, index in self.tokenizer.word2Index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            elif word == "<pad>":
                embedding_matrix[index] = [0] * 200
            else:
                missed.append(word)
                embedding_matrix[index] = [1] * 200
        print("Length embedding matrix: " + str(len(embedding_matrix)))
        print("Words not embedded: " + str(len(missed)))
        return embedding_matrix

    # reviews should be a list of list of words in reviews
    def load_ELMO(self, module,reviews):
        elmo = hub.load(module)
        embeddings = elmo.signatures['default'](reviews)
        with tf.compat.v1.Session() as sess:
            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
            x = sess.run(embeddings)
            return x
        # with tf.Graph().as_default():
        #     sentences = tf.compat.v1.placeholder(tf.string)
        #     embed = hub.load(module)
        #     embeddings = embed.signatures['default'](sentences)
        #     session = tf.compat.v1.train.MonitoredSession()
        # return lambda x: session.run(embeddings, {sentences: x})

    # uses skip-gram
    def load_word2vec(self, reviews, num_cores):
        return Word2Vec(reviews, min_count=1, size=200, workers=num_cores, window=2, sg=1)

if __name__ == "__main__":
    sys.path.insert(1, '../datasets/')
    from vocab_gen import Tokenizer
    from YelpDataset import YelpDataset
    t = Tokenizer("yo", "../datasets/vocabulary.txt")
    em = Embedding(t)
    #switch between glove and conceptnet
    # em.load_embedding("glove.twitter.27B/glove.twitter.27B.200d.txt")
    # x = em.embed(200)
    # print(x)

    """ ELMO shit
    """

    # mkdir module/module_elmo2
    # curl -L "https://tfhub.dev/google/elmo/2?tf-hub-format=compressed" | tar -zxvC module/module_elmo2
    # em_elmo = Embedding(None)
    # x = em_elmo.load_ELMO("module/module_elmo2")
    # print(x(["yo name jeff"]))
    # print(x(["my jeff name yo"]))

    # em_w2c = Embedding(None)
    # with open('reviews.txt', 'r') as f:
    #     tokenized_reviews = [line.replace("'", "").replace(",","").rstrip('\n') for line in f]
    #     print(tokenized_reviews[0])
    #     embedded_elmo = em_w2c.load_ELMO("https://tfhub.dev/google/elmo/2",tf.convert_to_tensor(np.array(tokenized_reviews[:5])))
    #     print(embedded_elmo["default"])

    """ Tokenization of reviews and saving into txt file
    """

    # tokenized_reviews = []
    # yelp = YelpDataset("../datasets/yelp_review_training_dataset.jsonl")
    # print("# reviews: " + str(len(yelp.reviews)))
    # tokenized_reviews_with_ratings = []
    # for r in yelp.reviews:
    #     y = em.tokenizer.tokenize(r["input"])
    #     new_arr = []
    #     for i in y:
    #         new_arr.append([i, r["label"]])
    #     print(new_arr)
    #     tokenized_reviews_with_ratings.extend(new_arr)
    #     tokenized_reviews.append(y)
    # np.savetxt("review_ratings.txt",tokenized_reviews_with_ratings, fmt="%s")
    # np.savetxt("review_ratings.txt",tokenized_reviews, fmt="%s")

    """ Generating word2vec embeddings using Skip-Gram
    """

    # with open('reviews.txt', 'r') as f:
    #     print("tokenizing reviews...")
    #     tokenized_reviews = [[t.wordOrUnk( word.replace("'", "").rstrip('\n') ) for word in line.split(", ")] for line in f]
    #     # print(tokenized_reviews[0])
    #
    #     print("creating word embeddings...")
    #     embedded = em.load_word2vec(tokenized_reviews, multiprocessing.cpu_count()//2)
    #
    #     print("saving word embeddings...")
    #     embedded.save("embedded.bin")
    #     print(embedded.most_similar("bad"))


    # embedded = Word2Vec.load("embedded.bin")
    # embedded.wv.save_word2vec_format("embedded.txt",binary=False)
    # print(embedded.most_similar("pubewar"))

    em.load_embedding("embedded.txt")
    vocab_embedded = em.embed(200)
    print(vocab_embedded)
