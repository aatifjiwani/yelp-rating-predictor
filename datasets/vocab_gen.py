import os
import jsonlines
from tqdm import tqdm
import string
import re

import nltk
from nltk import RegexpTokenizer
from nltk.stem import SnowballStemmer
# nltk.download('punkt')

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import RobertaTokenizer, RobertaForMaskedLM
from segtok import tokenizer

from collections import Counter


PAD_TOKEN = 0
UNK_TOKEN = 1
class Tokenizer():

    def __init__(self, name, vocabulary_file):
        self.name = name
        self.word2Index = {}
        with open(vocabulary_file) as vcb:
            for line in vcb:
                word, idx = line.strip().split(None, 1) 
                self.word2Index[word] = int(idx)

        self.index2word = {v:k for k,v in self.word2Index.items()}
        self.stemmer = SnowballStemmer("english")
    
    def tokenize(self, review, should_stem = True):
        cleaned_review = clean_sentence(review.lower())

        if should_stem:
            cleaned_review = " ".join([self.stemmer.stem(word) for word in cleaned_review.split()])

        tokenized_review = tokenizer.word_tokenizer(cleaned_review)

        return tokenized_review

    def tokenize2Index(self, review, should_stem = True):
        tokenized_review = self.tokenize(review, should_stem)

        tokens2indices = []
        for tokenizedWord in tokenized_review:
            tokens2indices.append(self.word2Index.get(tokenizedWord, UNK_TOKEN))

        return tokens2indices

    def wordOrUnk(self, word):
        return word if word in self.word2Index else "<unk>"

    def index2words(self, logits):
        pass

class ByteBPETokenizer:
    def __init__(self, vocab_json, merge_txt, max_length=750):
        self.tokenizer = ByteLevelBPETokenizer(vocab_json, merge_txt)
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(max_length=max_length)
        self.tokenizer.add_special_tokens(["[PAD]", "[CLS]"])
        # self.tokenizer.post_processor = RobertaProcessing(("</s>", 2), ("<s>", 1))
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def encode(self, review):
        review = clean_sentence(review)
        encoded = self.tokenizer.encode(review.lower())
        # pp_encoded = self.tokenizer.post_process(encoded)
        return encoded

    def tokenize2Index(self, review, should_stem=False):
        encoded = self.encode(review)

        return encoded.ids

    def trainBPE(self, paths, vocab_size=30000, min_frequency=10, special_tokens=["[PAD]", "[CLS]"]):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
        tokenizer.save("yelp_bpe/", "yelp-bpe")


class VocabularyGenerator():

    def __init__(self, src_jsonl_file, dest_txt_file):
        self.src = src_jsonl_file
        self.dest = dest_txt_file
        self.words = Counter()
        

    def parse_words(self):
        stemmer = SnowballStemmer("english")
        with jsonlines.open(self.src) as reader:
            for obj in tqdm(reader.iter(type=dict, skip_invalid=True)):
                # review = tokenizer.word_tokenizer(obj["text"].lower())
                # curr_words += review
                review = obj["text"]
                cleaned_review = clean_sentence(review)
                
                # stemmed_review = " ".join([stemmer.stem(word) for word in cleaned_review.split()]) UNCOMMENT THIS LINE FOR STEMMING
                tokenized_review = tokenizer.word_tokenizer(cleaned_review.lower())
                self.words.update(tokenized_review)
                
        print(len(self.words), "unique total words")

    def save_vocabulary(self):
        with open(self.dest, "w+") as f:
            f.write("<pad> 0\n")
            f.write("<unk> 1\n")

            i = 2
            sorted_tokens = sorted(self.words, key=lambda x: -self.words[x])
            if (len(sorted_tokens) > 50000):
                sorted_tokens = sorted_tokens[:50000]

            for word in tqdm(sorted_tokens):
                f.write("{} {}\n".format(word, i))
                i += 1

        print("finished creating vocabulary.")

def clean_sentence(sentence):
    ##removing websites
    sentence = re.sub(r"(http)?s?:?\/\/[A-Za-z0-9^,!.\/'+-=_?]+", "", sentence)

    #numbers
    sentence = re.sub(r"(\d+)(k)", r"\g<1> thousand", sentence)
    sentence = re.sub(r"(\d+)([a-zA-z]+)", r"\g<1> \g<2>", sentence)
    #convert numbers to words
    sentence = re.sub(r"1", " one ", sentence)
    sentence = re.sub(r"2", " two ", sentence)
    sentence = re.sub(r"3", " three ", sentence)
    sentence = re.sub(r"4", " four ", sentence)
    sentence = re.sub(r"5", " five ", sentence)
    sentence = re.sub(r"6", " six ", sentence)
    sentence = re.sub(r"7", " seven ", sentence)
    sentence = re.sub(r"8", " eight ", sentence)
    sentence = re.sub(r"9", " nine ", sentence)
    sentence = re.sub(r"0", " zero ", sentence)

    # removing extraneous symbols
    sentence = re.sub(r"[^A-Za-z0-9^,!.\/'+-=%]", " ", sentence)

    # expanding contraction
    sentence = re.sub(r"\'s", " is ", sentence)
    sentence = re.sub(r"\'ve", " have ", sentence)
    sentence = re.sub(r"n't", " not ", sentence)
    sentence = re.sub(r"i'm", " i am ", sentence)
    sentence = re.sub(r"\'re", " are ", sentence)
    sentence = re.sub(r"\'d", " would ", sentence)
    sentence = re.sub(r"\'ll", " will ", sentence)

    #spacing out symbols
    sentence = re.sub(r",", " ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\/", " ", sentence)
    sentence = re.sub(r"\^", " ^ ", sentence)
    sentence = re.sub(r"\+", " + ", sentence)
    sentence = re.sub(r"\-", " - ", sentence)
    sentence = re.sub(r"\=", " = ", sentence)
    sentence = re.sub(r"'", " ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r"%", " : ", sentence)

    return sentence

if __name__ == "__main__":
    # sent = "I have to say that this office really has it together, they are so organized and friendly!  Dr. J. Phillipp is a great dentist, very friendly and professional.  The dental assistants that helped in my procedure were amazing, Jewel and Bailey helped me to feel comfortable!  I don't have dental insurance, but they have this insurance through their office you can purchase for $80 something a year and this gave me 25% off all of my dental work, plus they helped me get signed up for care credit which I knew nothing about before this visit!  I highly recommend this office for the nice synergy the whole office has!"
    # tokenized = tokenizer.word_tokenizer(sent.lower())
    # print(tokenized)
    sent = "what's http://youtu.be/By-A7AN4jEA i've don't i'm you're i'd i'll we love Dr. B, Gibi and the entire Elite Family!!!!!! \
        \nThey all take such great care of our family!!!! Recommend scheduling your appointments soon!!\n\nThank you for all \
        you do for us.. Love you all.. 11th 111lbs 1123423am -blah .awdnw 'awdkawdn \awdawd 1234567891234"
    # v = VocabularyGenerator("yelp_review_training_dataset.jsonl", "vocabulary_nostem.txt")
    # review = token.tokenize(sent.lower())
    
    # cleaned = clean_sentence(sent)

    # print(sent)
    # print(cleaned)

    # stemmer = SnowballStemmer("english")
    # print(stemmer.stem("hygenist"))
    # print(" ".join([stemmer.stem(word) for word in cleaned.split()]))

    # print(review)
    # v.parse_words()
    # v.save_vocabulary()

    # t = Tokenizer("tokenizer", "vocabulary.txt")
    # print(len(t.word2Index))

    # print(clean_sentence("I really enjoyed my stay at this hotel on 5/12/2020!! This is jibberish abcd but here is this cool website: https://www.berkeley.edu"))
    sent = "I really enjoyed my stay at this hotel on five one two two zero two zero ! !  I am sure we will come again! Do not mind this jibberish abcd but here is this cool website :"
    stemmer = SnowballStemmer("english")
    cleaned_review = " ".join([stemmer.stem(word) for word in sent.split()])
    tokenized_review = tokenizer.word_tokenizer(cleaned_review)
    print(tokenized_review)
    # bpe = ByteBPETokenizer("yelp_bpe/yelp-bpe-vocab.json", "yelp_bpe/yelp-bpe-merges.txt")
    # enc = bpe.encode("I really enjoyed my stay at this hotel on 5/12/2020!! This is jibberish abcd but here is this cool website: https://www.berkeley.edu")
    # print(enc.tokens)
    # bpe.trainBPE(paths=["cleaned_reviews.txt"], vocab_size=25000)
    






    