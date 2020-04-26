from segtok import tokenizer
import os
import jsonlines
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk import RegexpTokenizer
import string
import re
from nltk.stem import SnowballStemmer

class Dictionary():
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def __init__(self, name):
        self.name = name

class VocabularyGenerator():

    def __init__(self, src_jsonl_file, dest_txt_file):
        self.src = src_jsonl_file
        self.dest = dest_txt_file
        self.words = None
        

    def parse_words(self):
        curr_words = []
        stemmer = SnowballStemmer("english")
        i = 0
        with jsonlines.open(self.src) as reader:
            for obj in tqdm(reader.iter(type=dict, skip_invalid=True)):
                # review = tokenizer.word_tokenizer(obj["text"].lower())
                # curr_words += review
                review = obj["text"]
                cleaned_review = clean_sentence(review)
                stemmed_review = " ".join([stemmer.stem(word) for word in cleaned_review.split()])
                tokenized_review = tokenizer.word_tokenizer(stemmed_review)
                curr_words += tokenized_review

                i += 1
                if (i > 100000):
                    break

        self.words = set(curr_words)
        print(len(self.words), "unique total words")

    def save_vocabulary(self):
        with open(self.dest, "w+") as f:
            f.write("<pad> 0\n")
            f.write("<unk> 1\n")

            i = 2
            sorted_tokens = list(self.words)
            sorted_tokens.sort()
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

    # removing extraneous symbols
    sentence = re.sub(r"[^A-Za-z0-9^,!.\/'+-=%]", " ", sentence)

    # expanding contraction
    sentence = re.sub(r"\'s", " is ", sentence)
    sentence = re.sub(r"\'ve", " have ", sentence)
    sentence = re.sub(r"n't", " not ", sentence)
    sentence = re.sub(r"i'm", "i am ", sentence)
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
        you do for us.. Love you all.. 11th 111lbs 1123423am -blah .awdnw 'awdkawdn \awdawd "
    v = VocabularyGenerator("yelp_review_training_dataset.jsonl", "vocabulary.txt")
    # review = token.tokenize(sent.lower())
    
    # cleaned = clean_sentence(sent)

    # print(sent)
    # print(cleaned)

    stemmer = SnowballStemmer("english")
    print(stemmer.stem("hygenist"))
    # print(" ".join([stemmer.stem(word) for word in cleaned.split()]))

    # print(review)
    # v.parse_words()
    # v.save_vocabulary()



    